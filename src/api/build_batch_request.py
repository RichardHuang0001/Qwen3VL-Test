import json
import yaml
from pathlib import Path
from tqdm import tqdm

# --- 辅助函数 1: 加载配置 (无修改) ---

def load_config(config_path="config.yaml") -> dict:
    """加载全局 YAML 配置文件"""
    root_dir = Path(__file__).parent.parent.parent
    config_file_path = root_dir / config_path
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_file_path}' 未找到。")
        exit(1)
    except Exception as e:
        print(f"加载 config.yaml 时出错: {e}")
        exit(1)

# --- 辅助函数 2: (核心) 封装Prompt (!! 已修改 - V3 增强版 !!) ---

def create_structured_prompt(base64_data_url: str, model_name: str) -> dict:
    """
    为单张图片创建结构化的API请求体 (body)。
    (V5: 基于提示词工程最佳实践 - 简化+少量示例+类别平衡)
    """
    
    # --- (V5: 优化的提示词设计 - 基于正确的提示词工程) ---
    prompt_text = (
        "任务：检测并分类图中的道路病害。\n\n"
        
        "【4种缺陷类型 - 快速判断表】\n"
        "| 类型 | 关键特征 | Label |\n"
        "|------|--------|-------|\n"
        "| 纵向裂缝 | 沿行车方向（前后）的线条，通常在轮迹上 | longitudinal_crack |\n"
        "| 横向裂缝 | 垂直行车方向（左右）的线条，横跨车道 | transverse_crack |\n"
        "| 网状裂缝 | 密集的多向交叉裂缝网络，不是平行线 | alligator_crack |\n"
        "| 坑洞 | 凹陷缺失，不是线条，内部深暗 | pothole |\n\n"
        
        "【易混淆对比】\n"
        "❌ 错误：所有裂缝都标为纵向 → ✅ 正确：注意横向和网状的细节\n"
        "❌ 错误：坑洞看起来像暗色线条 → ✅ 正确：坑洞是凹陷，不是线\n"
        "❌ 错误：几条平行裂缝当网状 → ✅ 正确：网状是多向交叉\n\n"
        
        "【多样化示例 - 覆盖所有类别】\n\n"
        "例子1（横向裂缝 - 注意方向！）：\n"
        "{\n"
        "  \"defect_type\": \"横向裂缝\",\n"
        "  \"detections\": [{\"label\": \"transverse_crack\", \"box_2d\": [0.25, 0.1, 0.35, 0.9]}]\n"
        "}\n\n"
        "例子2（网状裂缝 - 注意多向交叉！）：\n"
        "{\n"
        "  \"defect_type\": \"网状裂缝\",\n"
        "  \"detections\": [{\"label\": \"alligator_crack\", \"box_2d\": [0.1, 0.2, 0.9, 0.8]}]\n"
        "}\n\n"
        "例子3（多缺陷混合）：\n"
        "{\n"
        "  \"defect_type\": \"纵向裂缝\",\n"
        "  \"detections\": [\n"
        "    {\"label\": \"longitudinal_crack\", \"box_2d\": [0.2, 0.1, 0.8, 0.3]},\n"
        "    {\"label\": \"transverse_crack\", \"box_2d\": [0.3, 0.5, 0.5, 0.9]}\n"
        "  ]\n"
        "}\n\n"
        "例子4（坑洞 - 注意内部深暗，不是线）：\n"
        "{\n"
        "  \"defect_type\": \"坑洞\",\n"
        "  \"detections\": [{\"label\": \"pothole\", \"box_2d\": [0.4, 0.35, 0.6, 0.65]}]\n"
        "}\n\n"
        
        "【判断步骤】\n"
        "1️⃣  看是否有病害 → 有就继续，没有就返回 \"detections\": []\n"
        "2️⃣  问：多向交叉网络？ → 是 = 网状裂缝\n"
        "3️⃣  问：凹陷不是线？ → 是 = 坑洞\n"
        "4️⃣  问：左右方向？ → 是 = 横向裂缝\n"
        "5️⃣  其他 = 纵向裂缝\n\n"
        
        "【关键要求】\n"
        "✓ 每个检测都需要正确的英文标签（不能全是 'crack'）\n"
        "✓ 如果看到多个缺陷，列出所有实例\n"
        "✓ 坐标范围 0.0-1.0（相对坐标）\n"
        "✓ 只返回 JSON，不要其他文字\n\n"
        
        "【JSON格式】\n"
        "{\n"
        "  \"defect_type\": \"<中文类型>\",\n"
        "  \"detections\": [{\"label\": \"<label>\", \"box_2d\": [top, left, bottom, right]}]\n"
        "}"
    )
    # --- (V5 修改结束) ---

    # 返回API "body" 字段所需的完整结构 (无修改)
    return {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "你是一个专业的、经验丰富的土木工程病害检测专家。"
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": base64_data_url
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ]
            }
        ]
    }

# --- 主函数 (无修改) ---

def main():
    """
    主函数：加载Base64数据，封装Prompt，并写入 .jsonl 文件。
    """
    print("--- 开始构建批量API请求文件 (V5 - 优化版：基于提示词工程最佳实践) ---")

    # 1. 加载配置
    config = load_config()
    root_dir = Path(__file__).parent.parent.parent
    
    model_name = config.get('model_name', 'qwen-vl-plus') 
    processed_dir = root_dir / config['data']['processed_dir']
    
    input_file = processed_dir / "preprocessed_images.json"
    output_file = processed_dir / "batch_tasks_input.jsonl"

    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"使用模型: {model_name}")

    # 2. 加载 preprocessed_images.json
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            base64_data_store = json.load(f)
    except FileNotFoundError:
        print(f"\n错误： {input_file} 未找到。")
        print("请先运行 src/data/preprocess.py")
        return
    except Exception as e:
        print(f"加载 {input_file} 时出错: {e}")
        return

    if not base64_data_store:
        print("错误： preprocessed_images.json 文件为空。")
        return
        
    print(f"\n已加载 {len(base64_data_store)} 张图片的Base64数据。开始封装Prompt...")

    # 3. 遍历、封装并写入 .jsonl 文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for image_id, base64_url in tqdm(base64_data_store.items(), desc="封装请求"):
                
                # 4. 获取封装好的请求体 (body)
                request_body = create_structured_prompt(base64_url, model_name)
                
                # 5. 组装成批量API要求的 *完整* 格式
                batch_task = {
                    "custom_id": image_id,
                    "method": "POST",  
                    "url": "/v1/chat/completions", 
                    "body": request_body 
                }
                
                # 6. 将该任务作为一行JSON写入文件
                f.write(json.dumps(batch_task) + '\n')

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        return

    print(f"\n--- 构建完毕 ---")
    print(f"成功！已将 {len(base64_data_store)} 个API任务写入 {output_file}")
    print("\n下一步：请运行 src/api/submit_batch_job.py 来提交您的实验。")

if __name__ == "__main__":
    main()
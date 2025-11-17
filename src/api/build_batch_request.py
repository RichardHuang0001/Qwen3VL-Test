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

# --- 辅助函数 2: (核心) 封装Prompt (!! 已修改 !!) ---

def create_structured_prompt(base64_data_url: str, model_name: str) -> dict:
    """
    为单张图片创建结构化的API请求体 (body)。
    (V2: 要求相对坐标)
    """
    
    # --- (开始修改) ---
    # 我们将Prompt更新为强制要求0.0到1.0之间的相对坐标
    prompt_text = (
        "请仔细检查这张图片。\n"
        "1. 首先，识别这张图片中包含的主要病害类型（例如：纵向裂缝、横向裂缝、网状裂缝、坑洞、锈迹、或无病害）。\n"
        "2. 然后，找出图中所有的'裂缝'实例。\n"
        "3. (重要) 最后，请以JSON格式返回你的分析结果。在 'box_2d' 字段中, 你 *必须* 提供 **相对坐标** (即 0.0 到 1.0 之间的小数)。\n"
        "   - 'ymin' 必须是 (裂缝的最小y像素 / 图片总高度)\n"
        "   - 'xmin' 必须是 (裂缝的最小x像素 / 图片总宽度)\n"
        "   - (以此类推...)\n"
        "请严格按照以下JSON格式返回，不要添加任何额外的解释性文字：\n\n"
        "{\n"
        "  \"defect_type\": \"[此处填写病害类型]\",\n"
        "  \"detections\": [\n"
        "    {\"label\": \"crack\", \"box_2d\": [ymin_relative, xmin_relative, ymax_relative, xmax_relative]},\n"
        "    {\"label\": \"crack\", \"box_2d\": [ymin_relative, xmin_relative, ymax_relative, xmax_relative]}\n"
        "  ]\n"
        "}\n\n"
        "如果未检测到任何裂缝，请返回一个空的detections列表 []。"
    )
    # --- (修改结束) ---

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
    print("--- 开始构建批量API请求文件 (V2 - 相对坐标版) ---") # (我只改了这里的打印信息)

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
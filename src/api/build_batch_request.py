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
    (V3: 增加类别先验知识描述，提升检测准确率)
    """
    
    # --- (V3: 嵌入类别先验知识) ---
    prompt_text = (
        "请仔细检查这张道路图片，识别其中的病害类型并定位病害位置。\n\n"
        
        "【背景知识】道路病害主要包括以下4种类型：\n"
        "1. **纵向裂缝 (D00 - Longitudinal Crack)**\n"
        "   - 特征：沿车辆行驶方向延伸的细长裂缝，通常出现在轮迹带位置，形态笔直或略弯\n"
        "   - 形态：线状，顺着道路纵向延展，与道路轴线平行\n"
        "   - 成因：温度收缩、基层不均匀沉降或沥青层疲劳损伤\n\n"
        
        "2. **横向裂缝 (D10 - Transverse Crack)**\n"
        "   - 特征：方向大致垂直于行车方向，多为单条横断线或间隔分布\n"
        "   - 形态：横跨车道，宽度较均匀，切断路面表层\n"
        "   - 成因：温度变化或收缩应力，常见于气温变化较大的地区\n\n"
        
        "3. **网状裂缝 (D20 - Alligator Crack)**\n"
        "   - 特征：呈网状交叉分布，类似龟壳或鳄鱼皮花纹\n"
        "   - 形态：多条裂缝互相连接形成多边形块体，密集交叉\n"
        "   - 成因：路面疲劳破坏，通常预示基层失效，严重时伴随剥落\n\n"
        
        "4. **坑槽 (D40 - Pothole)**\n"
        "   - 特征：路面表层材料脱落后形成的明显凹坑，边缘不规则\n"
        "   - 形态：破损区域内部较深，常积水或呈暗色，周围可见松散碎石\n"
        "   - 成因：疲劳破坏、雨水侵蚀或冻融循环，是严重的路面损坏\n\n"
        
        "【检测任务】\n"
        "1. 识别图片中的主要病害类型（从上述4类中选择，或标注为'无病害'）\n"
        "2. 找出图中所有病害实例，并标注其边界框位置\n"
        "3. 以JSON格式返回检测结果\n\n"
        
        "【坐标要求】\n"
        "边界框必须使用 **相对坐标**（0.0 到 1.0 之间的小数）：\n"
        "   - ymin = 病害区域最小y像素 / 图片总高度\n"
        "   - xmin = 病害区域最小x像素 / 图片总宽度\n"
        "   - ymax = 病害区域最大y像素 / 图片总高度\n"
        "   - xmax = 病害区域最大x像素 / 图片总宽度\n\n"
        
        "【输出格式】\n"
        "严格按照以下JSON格式返回，不要添加任何额外的解释性文字：\n\n"
        "{\n"
        "  \"defect_type\": \"纵向裂缝\",\n"
        "  \"detections\": [\n"
        "    {\n"
        "      \"label\": \"longitudinal_crack\",\n"
        "      \"box_2d\": [0.23, 0.15, 0.78, 0.92]\n"
        "    },\n"
        "    {\n"
        "      \"label\": \"longitudinal_crack\",\n"
        "      \"box_2d\": [0.10, 0.45, 0.35, 0.88]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "【字段说明】\n"
        "- defect_type: 主要病害类型的中文名称（纵向裂缝/横向裂缝/网状裂缝/坑槽/无病害）\n"
        "- label: 检测实例的英文标识符，必须使用以下之一：\n"
        "  * longitudinal_crack (纵向裂缝)\n"
        "  * transverse_crack (横向裂缝)\n"
        "  * alligator_crack (网状裂缝)\n"
        "  * pothole (坑槽)\n"
        "- box_2d: 边界框的相对坐标 [ymin, xmin, ymax, xmax]，数值范围 0.0-1.0\n\n"
        "注意：\n"
        "- 如果未检测到任何病害，detections 应为空列表 []\n"
        "- 每个检测实例都必须提供准确的边界框坐标\n"
        "- 请根据上述背景知识准确判断病害类型"
    )
    # --- (V3 修改结束) ---

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
    print("--- 开始构建批量API请求文件 (V3 - 增强版：嵌入类别先验知识) ---")

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
import json
import yaml
import re
import cv2  # (!! 导入新安装的 OpenCV !!)
from pathlib import Path
from tqdm import tqdm

# --- 辅助函数 1: 加载配置 ---
def load_config(config_path="config.yaml") -> dict:
    """加载全局 YAML 配置文件"""
    # 脚本路径: src/analysis/visualize_results.py
    # 根目录: .parent.parent.parent
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

# --- 辅助函数 2: (鲁棒) 查找原始图片 ---
def find_original_image(base_path_with_no_ext: Path):
    """
    根据 custom_id (无扩展名) 查找 .jpg, .png, 或 .jpeg 格式的原始图片
    """
    for ext in ['.jpg', '.png', '.jpeg']:
        image_path = base_path_with_no_ext.with_suffix(ext)
        if image_path.exists():
            return image_path
    return None

# --- 辅助函数 3: (鲁棒) 解析模型的JSON输出 ---
def parse_model_content(content_str: str) -> dict | None:
    """
    从模型返回的 content 字符串中提取 JSON。
    它会处理 ```json ... ``` 标记。
    """
    # 使用正则表达式查找 ```json ... ``` 块
    match = re.search(r'```json\s*(\{.*?\})\s*```', content_str, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        # 如果没有找到 ```json ... ```, 假设整个字符串就是JSON
        # (这很脆弱，但作为后备)
        json_str = content_str.strip()
        
    try:
        # 将转义字符（如 \n）替换
        json_str = json_str.replace(r'\n', '\n')
        return json.loads(json_str)
    except json.JSONDecodeError:
        print(f"警告：无法解析模型返回的JSON: {content_str[:50]}...")
        return None

# --- 主函数 ---
def main():
    print("--- 启动结果可视化脚本 ---")
    
    # 1. 加载配置
    config = load_config()
    root_dir = Path(__file__).parent.parent.parent
    
    # 定义关键路径
    raw_image_dir = root_dir / config['data']['raw_dir']
    results_dir = root_dir / config['results']['output_dir']
    
    # (!!) 自动查找最新的结果文件 (带时间戳的)
    result_files = list(results_dir.glob("api_raw_results*.jsonl"))
    if not result_files:
        print(f"\n[!!] 错误：在 {results_dir} 中未找到任何 'api_raw_results...' 文件。")
        return

    # 按修改时间排序，找到最新的一个
    latest_result_file = max(result_files, key=lambda p: p.stat().st_mtime)
    print(f"\n[信息] 正在处理最新的结果文件:\n  > {latest_result_file.name}")

    # 定义输出目录
    output_viz_dir = results_dir / "visualizations"
    output_viz_dir.mkdir(exist_ok=True)
    print(f"[信息] 可视化结果将保存到:\n  > {output_viz_dir}")

    # 2. 逐行读取最新的结果文件
    try:
        with open(latest_result_file, 'r', encoding='utf-8') as f:
            # 计算总行数以便tqdm显示进度
            lines = f.readlines()
            total_lines = len(lines)
            print(f"  > 共有 {total_lines} 条结果需要处理。")

            for line in tqdm(lines, desc="生成可视化"):
                try:
                    data = json.loads(line)
                    
                    # 3. 检查API调用是否成功
                    if data.get("error") is not None or "response" not in data:
                        continue # 跳过失败的API调用
                        
                    # 4. 提取关键信息
                    custom_id = data["custom_id"] # (例如 "China MotorBike/China_MotorBike_000042")
                    content_str = data["response"]["body"]["choices"][0]["message"]["content"]
                    
                    # 5. (!! 核心) 查找并加载原始图片
                    original_image_path = find_original_image(raw_image_dir / custom_id)
                    if not original_image_path:
                        print(f"警告：未找到 {custom_id} 对应的原始图片，跳过...")
                        continue
                    
                    # 使用 OpenCV 读取图片
                    image = cv2.imread(str(original_image_path))
                    img_h, img_w, _ = image.shape
                    
                    # 6. 解析模型的JSON输出
                    model_output = parse_model_content(content_str)
                    if not model_output:
                        continue # 解析失败，跳过

                    # 7. (!! 核心) 绘制 BoundingBox 和标签
                    
                    # 绘制模型识别的总体类型
                    defect_type = model_output.get("defect_type", "N/A")
                    cv2.putText(image, f"Type: {defect_type}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) # 红色

                    detections = model_output.get("detections", [])
                    for det in detections:
                        box = det.get("box_2d")
                        if not box or len(box) != 4:
                            continue # 跳过格式错误的 box

                        # (!! 关键的坐标转换 !!)
                        # 我们假设模型返回 [ymin_rel, xmin_rel, ymax_rel, xmax_rel]
                        try:
                            ymin = int(float(box[0]) * img_h)
                            xmin = int(float(box[1]) * img_w)
                            ymax = int(float(box[2]) * img_h)
                            xmax = int(float(box[3]) * img_w)
                        except (ValueError, TypeError):
                            print(f"警告: {custom_id} 的坐标格式错误: {box}，跳过...")
                            continue

                        label = det.get("label", "N/A")

                        # 绘制矩形 (绿色)
                        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                        
                        # 绘制标签 (绿色)
                        cv2.putText(image, label, (xmin, ymin - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    # 8. 保存已绘制的图片
                    # (我们保留子目录结构，例如 "results/visualizations/China MotorBike/...")
                    output_image_path = output_viz_dir / f"{custom_id}.jpg"
                    
                    # 确保子目录存在
                    output_image_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    cv2.imwrite(str(output_image_path), image)

                except Exception as e:
                    print(f"处理行 {line[:30]}... 时发生未知错误: {e}")

    except Exception as e:
        print(f"\n[!!] 致命错误: 无法读取结果文件 {latest_result_file}。错误: {e}")
        return

    print(f"\n--- 可视化完毕 ---")
    print(f"所有图片已保存到 {output_viz_dir}")

if __name__ == "__main__":
    main()
    
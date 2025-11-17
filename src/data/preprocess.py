import base64
import json
import mimetypes
import os
import yaml # 导入 PyYAML
from pathlib import Path
from tqdm import tqdm

# 定义支持的图片文件扩展名
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

# --- 辅助函数 ---

def load_config(config_path="config.yaml") -> dict:
    """加载全局 YAML 配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_path}' 未找到。")
        print("请确保您已在项目根目录创建了 config.yaml。")
        exit(1)
    except Exception as e:
        print(f"加载 config.yaml 时出错: {e}")
        exit(1)

def encode_image_to_base64_url(image_path: Path) -> str:
    """读取一张图片，将其编码为 Base64 Data URL"""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "application/octet-stream"

    with open(image_path, "rb") as image_file:
        binary_data = image_file.read()

    base64_encoded_data = base64.b64encode(binary_data).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

# --- 主函数 ---

def main():
    """
    主函数：遍历图片，仅执行Base64编码，并存入一个JSON文件。
    """
    print("--- 开始轻量级预处理 (仅Base64编码) ---")
    
    # 1. 加载配置
    config = load_config()
    
    # 从配置中获取路径
    root_dir = Path(__file__).parent.parent.parent  # <--- 这是修正后的一行
    input_dir = root_dir / config['data']['raw_dir']
    output_dir = root_dir / config['data']['processed_dir']
    
    # 定义预处理结果的输出文件
    output_file = output_dir / "preprocessed_images.json"

    # 确保输出目录存在
    output_dir.mkdir(exist_ok=True)

    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")

    if not input_dir.exists():
        print(f"\n错误：输入目录 {input_dir} 不存在。")
        print("请先将您的图片数据集放入 'data/raw' 文件夹中。")
        return

    # 2. 递归查找所有图片文件
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(input_dir.rglob(f'*{ext}'))

    if not image_paths:
        print(f"\n错误：在 {input_dir} 中未找到任何图片文件。")
        return

    print(f"\n找到了 {len(image_paths)} 张图片。开始进行Base64编码...")

    # 3. 遍历、编码并存入字典
    base64_data_store = {}
    # (在 main 函数的 try 循环内部)
    
    try:
        for image_path in tqdm(image_paths, desc="编码中"):
            
            # --- (开始修复) ---
            # 我们需要一个唯一的ID。
            # 使用相对于 input_dir 的路径来确保唯一性。
            relative_path = image_path.relative_to(input_dir)
            
            # 将 'D00_cracks/image_001.jpg' 转换为 'D00_cracks/image_001'
            # (在Windows上, 路径可能是 D00_cracks\image_001, 我们统一用 /)
            image_id = str(relative_path.with_suffix('')).replace(os.path.sep, '/')
            # --- (修复结束) ---

            
            base64_url = encode_image_to_base64_url(image_path)
            base64_data_store[image_id] = base64_url

    except Exception as e:
        print(f"\n处理过程中发生错误: {e}")
        return

    # 4. 将包含所有Base64数据的大字典一次性写入JSON文件
    print(f"\n编码完成。正在将结果写入 {output_file} ...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(base64_data_store, f, indent=4) # 存为格式化的JSON

    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\n--- 预处理完毕 ---")
    print(f"成功！已将 {len(base64_data_store)} 张图片的Base64数据存入 {output_file}")
    print(f"输出文件总大小: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    main()
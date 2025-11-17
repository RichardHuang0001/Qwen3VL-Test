import json
import yaml
from pathlib import Path

# --- 辅助函数 ---
# (这个函数和 preprocess.py 中的一样)
def load_config(config_path="config.yaml") -> dict:
    """加载全局 YAML 配置文件"""
    # 我们需要找到根目录来加载 config.yaml
    # __file__ -> src/data/check_data.py
    # .parent -> src/data
    # .parent.parent -> src
    # .parent.parent.parent -> Qwen3VL-Test1 (项目根目录)
    root_dir = Path(__file__).parent.parent.parent
    config_file_path = root_dir / config_path
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_file_path}' 未找到。")
        print("请确保您已在项目根目录创建了 config.yaml。")
        exit(1)
    except Exception as e:
        print(f"加载 config.yaml 时出错: {e}")
        exit(1)

# --- 主函数 ---

def main():
    """
    主函数：加载 preprocessed_images.json 并检查每张图片的大小。
    """
    print("--- 开始数据检查 (检查单张图片大小) ---")

    # 1. 定义我们关心的技术限制（单次API请求的负载大小）
    # 这是一个安全估值，6MB 是一个常见的API网关限制
    REQUEST_SIZE_LIMIT_MB = 6.0 
    
    # 2. 加载配置
    config = load_config()
    
    # 从配置中获取路径
    root_dir = Path(__file__).parent.parent.parent
    processed_dir = root_dir / config['data']['processed_dir']
    
    # 要检查的输入文件
    input_file = processed_dir / "preprocessed_images.json"

    print(f"正在检查文件: {input_file}")
    print(f"检查标准：单张图片估算请求大小是否 > {REQUEST_SIZE_LIMIT_MB} MB")

    if not input_file.exists():
        print(f"\n错误：文件 {input_file} 不存在。")
        print("请先运行 src/data/preprocess.py 来生成该文件。")
        return

    # 3. 加载预处理好的数据
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"加载 {input_file} 时出错: {e}")
        return
        
    if not data:
        print("错误： preprocessed_images.json 文件为空。")
        return
        
    print(f"\n已加载 {len(data)} 张图片的Base64数据。开始分析...")

    # 4. 遍历、检查并统计
    image_sizes = [] # 存储 (size_mb, image_id)
    oversized_images = []

    for image_id, base64_url in data.items():
        # 估算大小：获取base64字符串的字节数
        # 这是最接近API请求中该图片所占负载大小的值
        size_bytes = len(base64_url.encode('utf-8'))
        size_mb = size_bytes / (1024 * 1024)
        
        image_sizes.append((size_mb, image_id))
        
        # 检查是否超过限制
        if size_mb > REQUEST_SIZE_LIMIT_MB:
            oversized_images.append((image_id, size_mb))

    # 5. 报告统计结果
    if not image_sizes:
        print("未找到任何图片数据进行统计。")
        return

    # 计算统计数据
    total_images = len(image_sizes)
    max_size_mb, max_image_id = max(image_sizes)
    min_size_mb, min_image_id = min(image_sizes)
    avg_size_mb = sum([size for size, _ in image_sizes]) / total_images

    print("\n--- 检查完毕 ---")
    print(f"📊 统计总览：")
    print(f"  - 总计图片数: {total_images}")
    print(f"  - 平均大小: {avg_size_mb:.2f} MB")
    print(f"  - (最大) {max_image_id}.jpg (大小: {max_size_mb:.2f} MB)")
    print(f"  - (最小) {min_image_id}.jpg (大小: {min_size_mb:.2f} MB)")

    # 6. 报告风险
    print(f"\n🚨 风险报告 (阈值 = {REQUEST_SIZE_LIMIT_MB} MB)：")
    if not oversized_images:
        print(f"  - (好消息) 所有 {total_images} 张图片的估算请求大小均 < {REQUEST_SIZE_LIMIT_MB} MB。")
        print("  - 风险低。可以安全进入下一步。")
    else:
        print(f"  - (!! 警告 !!) 发现 {len(oversized_images)} 张图片可能超出 {REQUEST_SIZE_LIMIT_MB} MB 的API请求限制：")
        for image_id, size_mb in oversized_images:
            print(f"    - {image_id}.jpg (大小: {size_mb:.2f} MB)")
        print(f"\n  - 建议：在进入下一步（构建API请求）之前，请考虑压缩这些图片的分辨率并重新运行 preprocess.py。")

if __name__ == "__main__":
    main()
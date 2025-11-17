import sys
import yaml
import xml.etree.ElementTree as ET # (!! 核心: Python内置的XML解析器 !!)
from pathlib import Path
from collections import Counter # (!! 核心: 用于计数的最佳工具 !!)
from tqdm import tqdm

# --- 辅助函数 1: 加载配置 ---
def load_config(config_path="config.yaml") -> dict:
    """加载全局 YAML 配置文件"""
    # 脚本路径: src/data/explore_annotations.py
    # 根目录: .parent.parent.parent
    root_dir = Path(__file__).parent.parent.parent
    config_file_path = root_dir / config_path
    
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"错误：配置文件 '{config_file_path}' 未找到。")
        sys.exit(1)
    except Exception as e:
        print(f"加载 config.yaml 时出错: {e}")
        sys.exit(1)

# --- 主函数 ---
def main():
    print("--- 启动数据集(XML)探索脚本 ---")
    
    # 1. 加载配置
    config = load_config()
    root_dir = Path(__file__).parent.parent.parent
    raw_data_dir = root_dir / config['data']['raw_dir']

    print(f"将搜索目录: {raw_data_dir}")

    if not raw_data_dir.exists():
        print(f"\n[!!] 错误：数据目录 {raw_data_dir} 不存在。")
        sys.exit(1)

    # 2. 递归查找所有 .xml 标注文件
    xml_files = list(raw_data_dir.rglob('*.xml'))

    if not xml_files:
        print(f"\n[!!] 错误：在 {raw_data_dir} 中未找到任何 .xml 文件。")
        sys.exit(1)
        
    print(f"\n[信息] 找到了 {len(xml_files)} 个 XML 标注文件。开始解析...")

    # 3. (!! 核心) 遍历、解析并计数
    
    # 我们使用 Counter 来自动计数
    class_counter = Counter()
    total_objects = 0

    try:
        for xml_path in tqdm(xml_files, desc="解析XML中"):
            try:
                # 解析XML文件
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # 查找所有的 <object> 标签
                for obj in root.findall('object'):
                    # 找到 <name> 标签
                    name_element = obj.find('name')
                    if name_element is not None:
                        class_name = name_element.text
                        
                        # (!! 计数 !!)
                        class_counter[class_name] += 1
                        total_objects += 1
                        
            except ET.ParseError:
                print(f"\n警告：无法解析文件 {xml_path} (可能已损坏)，已跳过。")
            except Exception as e:
                print(f"\n处理 {xml_path} 时发生未知错误: {e}")

    except KeyboardInterrupt:
        print("\n[中断] 用户手动停止。")
        sys.exit(0)

    # 4. (!! 核心) 打印统计报告
    print("\n--- 探索完毕：数据集统计报告 ---")
    
    if not class_counter:
        print("未在任何XML文件中找到 <object>/<name> 标签。")
        return

    print(f"  - 总计 XML 文件数: {len(xml_files)}")
    print(f"  - 总计 <object> 标注数: {total_objects}")
    print(f"  - 总计 唯一类别数: {len(class_counter)}")

    print("\n📊 类别分布 (按数量排序):")
    # .most_common() 会自动排序
    for class_name, count in class_counter.most_common():
        print(f"  - {class_name:<20} : {count} 个") # (左对齐，20个字符宽度)

    # 5. (!! 核心) 给出科研建议
    unique_classes = list(class_counter.keys())
    print("\n--- [!!] 科研建议 ---")
    print("您在 `data/raw/` 数据集中的真实类别是：")
    print(f"  {unique_classes}")
    print("\n在您下一次 (第三次) 实验中，您 *必须* 更新")
    print("`src/api/build_batch_request.py` 脚本中的 Prompt，")
    print(f"让模型去识别 *这些* 类别, 而不是只寻找 'crack'。")

if __name__ == "__main__":
    main()
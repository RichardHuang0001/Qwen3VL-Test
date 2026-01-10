"""
评估模型检测准确率
计算 Precision, Recall, F1-Score, mAP 等指标
"""

import json
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import re

# ============================
# 📌 配置区域：选择要评估的结果文件
# ============================

# 选项1: 自动使用最新的结果文件（默认）
USE_LATEST_RESULT = True

# 选项2: 手动指定结果文件名（当 USE_LATEST_RESULT = False 时生效）
# 示例: "api_raw_results_20260110_155552.jsonl"
SPECIFIC_RESULT_FILE = "api_raw_results_20260110_155552.jsonl"

# ============================

# --- 类别映射 ---
# XML标注中的类别 -> 模型输出的类别
# 支持V2版本（老）和V3版本（新）的模型输出
CLASS_MAPPING = {
    'D00': 'longitudinal_crack',  # 纵向裂缝
    'D10': 'transverse_crack',     # 横向裂缝
    'D20': 'alligator_crack',      # 网状裂缝
    'D40': 'pothole',              # 坑槽
    'Repair': 'repair'             # 修补区域
}

# 反向映射
REVERSE_CLASS_MAPPING = {v: k for k, v in CLASS_MAPPING.items()}

# 模型标签的预处理映射（处理V2版本的"crack"标签）
# 将旧版本的模糊标签映射到新版本的具体标签
LABEL_NORMALIZATION = {
    'crack': None,  # 旧版通用标签，将被跳过（因为无法确定具体类别）
    'longitudinal_crack': 'longitudinal_crack',
    'transverse_crack': 'transverse_crack',
    'alligator_crack': 'alligator_crack',
    'pothole': 'pothole',
    'repair': 'repair'
}

# --- 辅助函数 ---

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

def parse_xml_annotation(xml_path: Path) -> dict:
    """
    解析XML标注文件，提取Ground Truth
    返回: {'width': int, 'height': int, 'objects': [{'class': str, 'bbox': [xmin, ymin, xmax, ymax]}, ...]}
    """
    try:
        # 跳过macOS隐藏文件（以._开头的文件）
        if xml_path.name.startswith('._'):
            return None
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # 获取图片尺寸
        size = root.find('size')
        width = int(float(size.find('width').text))  # 支持小数坐标
        height = int(float(size.find('height').text))
        
        # 提取所有标注对象
        objects = []
        for obj in root.findall('object'):
            name = obj.find('name').text  # D00, D10, D20, D40
            bndbox = obj.find('bndbox')
            
            # 转换为整数，支持小数坐标
            bbox = [
                int(float(bndbox.find('xmin').text)),
                int(float(bndbox.find('ymin').text)),
                int(float(bndbox.find('xmax').text)),
                int(float(bndbox.find('ymax').text))
            ]
            
            objects.append({
                'class': name,
                'bbox': bbox
            })
        
        return {
            'width': width,
            'height': height,
            'objects': objects
        }
    except Exception as e:
        # 优化错误提示，跳过macOS隐藏文件的无用警告
        if not xml_path.name.startswith('._'):
            print(f"警告：解析 {xml_path.name} 失败: {e}")
        return None

def parse_model_content(content_str: str) -> dict:
    """
    从模型返回的 content 字符串中提取 JSON
    处理 ```json ... ``` 标记
    """
    match = re.search(r'```json\s*(\{.*?\})\s*```', content_str, re.DOTALL)
    
    if match:
        json_str = match.group(1)
    else:
        json_str = content_str.strip()
        
    try:
        json_str = json_str.replace(r'\n', '\n')
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def calculate_iou(box1, box2):
    """
    计算两个边界框的IoU (Intersection over Union)
    box1, box2: [xmin, ymin, xmax, ymax] (绝对像素坐标)
    """
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集
    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)
    
    if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
        return 0.0
    
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    
    # 计算并集
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def convert_relative_to_absolute(box_2d, img_width, img_height):
    """
    将相对坐标转换为绝对像素坐标
    box_2d: [ymin, xmin, ymax, xmax] (相对坐标 0-1)
    返回: [xmin, ymin, xmax, ymax] (绝对像素坐标)
    """
    ymin_rel, xmin_rel, ymax_rel, xmax_rel = box_2d
    
    xmin = int(xmin_rel * img_width)
    ymin = int(ymin_rel * img_height)
    xmax = int(xmax_rel * img_width)
    ymax = int(ymax_rel * img_height)
    
    return [xmin, ymin, xmax, ymax]

# --- 主评估函数 ---

def evaluate_detections(ground_truths, predictions, iou_threshold=0.5):
    """
    评估检测结果
    
    参数:
        ground_truths: {image_id: {'width': w, 'height': h, 'objects': [...]}}
        predictions: {image_id: {'width': w, 'height': h, 'detections': [...]}}
        iou_threshold: IoU阈值，默认0.5
    
    返回:
        metrics: 各类别的 precision, recall, f1, ap 等指标
    """
    # 按类别统计
    class_stats = defaultdict(lambda: {
        'tp': 0,  # True Positives
        'fp': 0,  # False Positives
        'fn': 0,  # False Negatives (ground truth未被检测到)
        'matched_pairs': []  # (预测置信度, IoU, 是否正确匹配)
    })
    
    # 遍历每张图片
    for image_id in tqdm(ground_truths.keys(), desc="评估检测结果"):
        gt_data = ground_truths.get(image_id)
        pred_data = predictions.get(image_id)
        
        if not gt_data:
            continue
        
        img_width = gt_data['width']
        img_height = gt_data['height']
        gt_objects = gt_data['objects']
        
        # 如果没有预测结果
        if not pred_data or not pred_data.get('detections'):
            # 所有ground truth都是false negatives
            for gt_obj in gt_objects:
                gt_class = CLASS_MAPPING.get(gt_obj['class'])
                if gt_class:
                    class_stats[gt_class]['fn'] += 1
            continue
        
        pred_detections = pred_data['detections']
        
        # 按类别分组
        gt_by_class = defaultdict(list)
        for gt_obj in gt_objects:
            gt_class_mapped = CLASS_MAPPING.get(gt_obj['class'])
            if gt_class_mapped:
                gt_by_class[gt_class_mapped].append(gt_obj['bbox'])
        
        pred_by_class = defaultdict(list)
        for pred in pred_detections:
            pred_class = pred.get('label')
            pred_box_rel = pred.get('box_2d')
            
            # 标签规范化处理（处理V2版本的"crack"标签）
            if pred_class in LABEL_NORMALIZATION:
                normalized_label = LABEL_NORMALIZATION[pred_class]
                # 跳过无法确定具体类别的标签
                if normalized_label is None:
                    continue
                pred_class = normalized_label
            
            if pred_class and pred_box_rel and len(pred_box_rel) == 4:
                # 转换为绝对坐标 [xmin, ymin, xmax, ymax]
                pred_box_abs = convert_relative_to_absolute(
                    pred_box_rel, img_width, img_height
                )
                pred_by_class[pred_class].append(pred_box_abs)
        
        # 对每个类别进行匹配
        for class_name in set(list(gt_by_class.keys()) + list(pred_by_class.keys())):
            gt_boxes = gt_by_class[class_name]
            pred_boxes = pred_by_class[class_name]
            
            matched_gt = set()  # 已匹配的ground truth索引
            
            # 对每个预测框
            for pred_box in pred_boxes:
                best_iou = 0
                best_gt_idx = -1
                
                # 找到最佳匹配的ground truth
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    iou = calculate_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # 判断是否匹配成功
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    class_stats[class_name]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_stats[class_name]['fp'] += 1
            
            # 未匹配的ground truth为false negatives
            unmatched_gt_count = len(gt_boxes) - len(matched_gt)
            class_stats[class_name]['fn'] += unmatched_gt_count
    
    # 计算各类别指标
    results = {}
    for class_name, stats in class_stats.items():
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[class_name] = {
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # 计算总体指标 (macro average)
    if results:
        avg_precision = sum(r['precision'] for r in results.values()) / len(results)
        avg_recall = sum(r['recall'] for r in results.values()) / len(results)
        avg_f1 = sum(r['f1'] for r in results.values()) / len(results)
        
        results['macro_average'] = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }
    
    return results

# --- 主函数 ---

def main():
    print("--- 启动检测结果评估脚本 ---\n")
    
    # 1. 加载配置
    config = load_config()
    root_dir = Path(__file__).parent.parent.parent
    
    raw_data_dir = root_dir / config['data']['raw_dir']
    results_dir = root_dir / config['results']['output_dir']
    
    # 2. 选择要评估的结果文件
    if USE_LATEST_RESULT:
        # 自动查找最新的结果文件
        result_files = list(results_dir.glob("api_raw_results*.jsonl"))
        if not result_files:
            print(f"错误：在 {results_dir} 中未找到任何结果文件。")
            return
        
        latest_result_file = max(result_files, key=lambda p: p.stat().st_mtime)
        print(f"[模式] 自动选择最新结果文件")
        print(f"[信息] 使用结果文件: {latest_result_file.name}\n")
    else:
        # 使用手动指定的结果文件
        latest_result_file = results_dir / SPECIFIC_RESULT_FILE
        if not latest_result_file.exists():
            print(f"错误：指定的结果文件 '{SPECIFIC_RESULT_FILE}' 不存在。")
            print(f"请检查路径：{latest_result_file}")
            return
        
        print(f"[模式] 使用指定的结果文件")
        print(f"[信息] 评估文件: {latest_result_file.name}\n")
    
    # 3. 加载Ground Truth (XML标注)
    print("[步骤 1/4] 加载Ground Truth标注...")
    xml_files = list(raw_data_dir.rglob('*.xml'))
    # 过滤掉macOS隐藏文件
    xml_files = [f for f in xml_files if not f.name.startswith('._')]
    
    ground_truths = {}
    for xml_path in tqdm(xml_files, desc="解析XML"):
        gt_data = parse_xml_annotation(xml_path)
        if gt_data and gt_data['objects']:  # 只保留有标注的图片
            # 使用相对路径作为image_id (去掉扩展名)
            image_id = str(xml_path.relative_to(raw_data_dir).with_suffix('').with_suffix(''))
            # 规范化路径分隔符
            image_id = image_id.replace('\\', '/')
            ground_truths[image_id] = gt_data
    
    print(f"加载了 {len(ground_truths)} 张有标注的图片\n")
    
    # 4. 加载Predictions (API结果)
    print("[步骤 2/4] 加载模型预测结果...")
    predictions = {}
    
    with open(latest_result_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in tqdm(lines, desc="解析预测"):
            try:
                data = json.loads(line)
                
                if data.get("error") or "response" not in data:
                    continue
                
                image_id = data["custom_id"]
                content_str = data["response"]["body"]["choices"][0]["message"]["content"]
                
                model_output = parse_model_content(content_str)
                if not model_output:
                    continue
                
                # 获取图片尺寸 (从ground truth中获取)
                if image_id in ground_truths:
                    predictions[image_id] = {
                        'width': ground_truths[image_id]['width'],
                        'height': ground_truths[image_id]['height'],
                        'detections': model_output.get('detections', [])
                    }
            
            except Exception as e:
                continue
    
    print(f"加载了 {len(predictions)} 张图片的预测结果\n")
    
    # 5. 评估
    print("[步骤 3/4] 计算评估指标...")
    iou_threshold = 0.5
    metrics = evaluate_detections(ground_truths, predictions, iou_threshold)
    
    # 6. 打印结果
    print(f"\n[步骤 4/4] 评估完成！(IoU阈值 = {iou_threshold})\n")
    print("=" * 80)
    print(f"{'类别':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'TP/FP/FN'}")
    print("=" * 80)
    
    # 类别名称映射（用于显示）
    class_display_names = {
        'longitudinal_crack': 'D00-纵向裂缝',
        'transverse_crack': 'D10-横向裂缝',
        'alligator_crack': 'D20-网状裂缝',
        'pothole': 'D40-坑槽'
    }
    
    for class_name in ['longitudinal_crack', 'transverse_crack', 'alligator_crack', 'pothole']:
        if class_name in metrics:
            m = metrics[class_name]
            display_name = class_display_names.get(class_name, class_name)
            print(f"{display_name:<25} {m['precision']:<12.3f} {m['recall']:<12.3f} "
                  f"{m['f1']:<12.3f} {m['tp']}/{m['fp']}/{m['fn']}")
        else:
            display_name = class_display_names.get(class_name, class_name)
            print(f"{display_name:<25} {'N/A':<12} {'N/A':<12} {'N/A':<12} 0/0/0")
    
    print("-" * 80)
    
    if 'macro_average' in metrics:
        m = metrics['macro_average']
        print(f"{'宏平均 (Macro Avg)':<25} {m['precision']:<12.3f} {m['recall']:<12.3f} "
              f"{m['f1']:<12.3f}")
    
    print("=" * 80)
    
    # 7. 保存详细结果到JSON
    output_file = results_dir / "evaluation_metrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n详细评估结果已保存到: {output_file}")
    print("\n--- 评估完成 ---")

if __name__ == "__main__":
    main()

#
# src/api/submit_batch_job.py (v3.1 - 鲁棒且带时间戳)
#
import sys
import time
import json
import yaml
import dashscope
from openai import OpenAI
from pathlib import Path

# --- 辅助函数: 加载配置 (无修改) ---
def load_config(config_path="config.yaml") -> dict:
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

# --- 辅助函数: 加载/保存任务状态 (无修改) ---
def load_job_state(state_file_path: Path) -> dict:
    if state_file_path.exists():
        try:
            with open(state_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_job_state(state_file_path: Path, state: dict):
    with open(state_file_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)

# --- 主函数 (鲁棒版 v3.1) ---
def main():
    print("--- 启动批量推理任务提交脚本 (v3.1 - 带时间戳版) ---")

    # 1. 加载配置和路径
    config = load_config()
    root_dir = Path(__file__).parent.parent.parent
    api_key = config.get('api_key')
    
    processed_dir = root_dir / config['data']['processed_dir']
    results_dir = root_dir / config['results']['output_dir']
    results_dir.mkdir(exist_ok=True) 

    input_file_path = processed_dir / "batch_tasks_input.jsonl"
    
    # (!!) 状态文件是固定的，这是我们唯一的“记忆”锚点
    job_state_file = results_dir / ".job_status.json"

    # 2. API Key 检查 (无修改)
    if not api_key or api_key == "sk-your-aliyun-api-key-here":
        print("\n[!!] 错误：API Key 未配置在 config.yaml 中。")
        sys.exit(1)
    
    dashscope.api_key = api_key
    client = OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 3. (!! 恢复逻辑 - 已修改 !!)
    job_id = None
    raw_output_file_path = None # 我们现在不知道输出路径
    state = load_job_state(job_state_file)
    
    if "job_id" in state:
        job_id = state["job_id"]
        
        # (!! 关键: 从状态中恢复文件名 !!)
        if "output_filename" not in state:
            print(f"[!!] 错误：缓存文件 {job_state_file.name} 已损坏，缺少 'output_filename'。")
            print("正在清理损坏的缓存，请重新提交新任务。")
            job_state_file.unlink()
            sys.exit(1)
            
        raw_output_file_path = Path(state["output_filename"]) # 重新转换为Path对象
        
        print(f"\n[恢复] 找到了一个正在运行的缓存任务: {job_id}")
        print(f"  > 目标文件: {raw_output_file_path.name}")
        print("将跳过“创建任务”步骤，直接开始“轮询”该任务...")

    # 4. (!! 创建新任务 - 已修改 !!)
    if job_id is None:
        print("\n[新任务] 未找到缓存任务。将创建一个新任务。")
        print("警告：此操作即将消耗API额度。")

        # (!! 关键: 生成带时间戳的新文件名 !!)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        raw_output_file_path = results_dir / f"api_raw_results_{timestamp}.jsonl"
        print(f"  > 本次运行的结果将保存到: {raw_output_file_path.name}")

        if not input_file_path.exists():
            print(f"\n[!!] 错误：任务文件 {input_file_path} 未找到。")
            print("请先运行 src/api/build_batch_request.py (确保是V2相对坐标版)。")
            sys.exit(1)
        
        try:
            # 步骤 4a: 上传文件
            print("  > 步骤 1/3: 正在上传任务文件...")
            with open(input_file_path, "rb") as f:
                file_object = client.files.create(file=f, purpose='batch')
            file_id = file_object.id
            print(f"  > 文件上传成功。File ID: {file_id}")

            # 步骤 4b: 创建任务
            print("  > 步骤 2/3: 正在创建批量推理任务...")
            job = client.batches.create(
                input_file_id=file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            job_id = job.id
            print(f"  > 任务创建成功。Job ID: {job_id}")

            # 步骤 4c: (!! 写入记忆 - 已修改 !!)
            print(f"  > 步骤 3/3: 正在将 Job ID 和文件名写入缓存 {job_state_file.name} ...")
            new_state = {
                "job_id": job_id,
                "file_id": file_id,
                "status": "created",
                "input_file": str(input_file_path),
                "output_filename": str(raw_output_file_path) # <-- (!!) 存入带时间戳的文件名
            }
            save_job_state(job_state_file, new_state)
            print("  > 缓存写入成功。")

        except Exception as e:
            print(f"\n[!!] 创建新任务时失败: {e}")
            print("请检查API Key、网络、文件以及'openai'库是否已安装。")
            sys.exit(1)

    # 5. 轮询循环 (无修改)
    print(f"\n[轮询] 开始监控 Job ID: {job_id} (每30秒查询一次)")
    job_status = None
    while True:
        try:
            job_status = client.batches.retrieve(job_id)
            status = job_status.status
            print(f"  > {time.strftime('%Y-%m-%d %H:%M:%S')} - 当前状态: {status}")

            if status == 'completed':
                print("\n[成功] 任务已成功完成！")
                break
            elif status in ['failed', 'cancelled']:
                print(f"\n[!!] 任务失败或被取消。")
                errors = getattr(job_status, 'errors', '无可用错误信息')
                print(f"错误详情: {errors}")
                if job_state_file.exists():
                    job_state_file.unlink()
                sys.exit(1)
            
            time.sleep(30)
        
        except KeyboardInterrupt:
            print("\n[中断] 检测到手动中断 (Ctrl+C)。")
            print(f"任务 {job_id} 仍在云端运行。下次运行此脚本可自动恢复。")
            sys.exit(0)
            
        except Exception as e:
            print(f"\n[!!] 轮询时发生网络错误: {e}")
            print("将在60秒后自动重试...")
            time.sleep(60)

    # 6. 下载结果 (无修改)
    # (变量 `raw_output_file_path` 已在步骤 3 或 4 中被正确设置)
    print("\n[下载] 正在下载原始结果文件...")
    try:
        output_file_id = job_status.output_file_id
        if not output_file_id:
            raise ValueError("API未返回 output_file_id，任务可能已完成但无输出。")
            
        content = client.files.content(output_file_id)
        
        with open(raw_output_file_path, "wb") as f:
            f.write(content.read())

        print(f"原始结果文件下载成功！已保存到:")
        print(f"{raw_output_file_path}") # (将打印出带时间戳的文件名)

    except Exception as e:
        print(f"\n[!!] 下载结果时失败: {e}")
        print(f"任务 {job_id} 已在云端完成。")
        print("请检查网络，然后 *重新运行此脚本*，它将自动重试下载。")
        sys.exit(1)

    # 7. 清理 (无修改)
    print("\n[清理] 任务已成功下载，正在清理缓存...")
    if job_state_file.exists():
        job_state_file.unlink()
    print(f"缓存文件 {job_state_file.name} 已删除。")
    print("\n--- 完整工作流执行完毕 ---")

if __name__ == "__main__":
    main()
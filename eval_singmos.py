import warnings
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd
import argparse
import subprocess
import time
import glob

def preprocess_pred_dir(pred_dir):
    """预处理 pred_dir：去掉 wav 文件名前缀 opencpop_ 和 acesinger_13#"""
    prefix_list = ["opencpop_", "acesinger_13#"]
    for wav_path in glob.glob(os.path.join(pred_dir, "*.wav")):
        fname = os.path.basename(wav_path)
        new_name = fname
        for prefix in prefix_list:
            if new_name.startswith(prefix):
                new_name = new_name[len(prefix):]
        if new_name != fname:
            new_path = os.path.join(pred_dir, new_name)
            if os.path.exists(new_path):
                os.remove(new_path)
            os.rename(wav_path, new_path)

def run_scorer(pred_scp, output_file):
    """调用 scorer.py 进行评分"""
    if os.path.exists(pred_scp) and os.path.getsize(pred_scp) == 0:
        print("⚠️ pred.scp 为空，无需评分。")
        return

    cmd = [
        "python", "../versa/versa/bin/scorer.py",
        "--score_config", "../versa/egs/separate_metrics/singmos.yaml",
        "--pred", pred_scp,
        "--output_file", output_file,
        "--io", "soundfile"
    ]
    # print(f"Running command: {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end='')
    process.wait()

def eval(pred_dir, output_dir):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    result_file = os.path.join(output_dir, "result.jsonl")

    # Step 1: 找到已完成的 key
    finished_keys = set()
    if os.path.exists(result_file):
        with open(result_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "key" in data:
                        finished_keys.add(os.path.splitext(data["key"])[0])
                except:
                    continue

    print(f"已完成 {len(finished_keys)} 条，将跳过这些样本。")

    # Step 2: 生成 pred.scp（只包含未完成的）
    pred_scp = os.path.join(output_dir, "pred.scp")
    with open(pred_scp, "w") as f:
        for wav_path in sorted(glob.glob(os.path.join(pred_dir, "*.wav"))):
            utt_id = os.path.splitext(os.path.basename(wav_path))[0]
            if utt_id in finished_keys:
                continue
            abs_path = os.path.abspath(wav_path)
            f.write(f"{utt_id} {abs_path}\n")

    # Step 3: 跑 scorer
    run_scorer(pred_scp, result_file)

    print(f"\n🎉 All done in {time.time() - start_time:.2f} seconds.")

def get_average_score(output_dir): 
    result_file = os.path.join(output_dir, "result.jsonl")
    if not os.path.exists(result_file):
        print("⚠️ 没有找到 result.jsonl")
        return

    records = []
    with open(result_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                file_id = os.path.splitext(data['key'])[0]
                if "singmos" in data:
                    records.append({
                        '文件ID': file_id,
                        'singmos': data['singmos']
                    })
            except json.JSONDecodeError:
                print(f"⚠️ 跳过无效行：{line}")

    # 除非调试，功能函数中不要打印结果！
    df = pd.DataFrame(records)
    if not df.empty:
        avg_score = df['singmos'].mean()
        success_count = len(df)
        return avg_score, success_count
    else:
        # 没有有效的评分记录。
        return 0, 0
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="预测音频所在路径")
    parser.add_argument("--output_dir", type=str, required=False, default="", help="输出结果路径，默认在 pred_dir 下创建 eval_singmos 文件夹")
    parser.add_argument("--no_eval", action='store_true', help="不评估，只计算平均分")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    if args.output_dir != "":
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(pred_dir, "eval_singmos")
    os.makedirs(output_dir, exist_ok=True)

    preprocess_pred_dir(pred_dir)

    if not args.no_eval:
        eval(pred_dir, output_dir)  # 单进程执行

    avg_score, success_count = get_average_score(output_dir)

    if success_count != 0:
        print(f"📊 SingMOS 平均值：{avg_score:.2f}")
        print(f"✅ 成功数: {success_count}")
    else:
        print("⚠️ 没有有效的评分记录。")
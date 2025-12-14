import warnings
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd
import argparse
import subprocess
import time
import glob
from multiprocessing import Pool

def run_scorer(args):
    pred_scp, output_file, part_id = args
    cmd = [
        "python", "../versa/versa/bin/scorer.py",
        "--score_config", "../versa/egs/separate_metrics/dnsmos.yaml",
        "--pred", pred_scp,
        "--output_file", output_file,
        "--io", "soundfile"
    ]
    print(f"[Part {part_id}] Running command: {' '.join(cmd)}\n")

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(f"[Part {part_id}] {line}", end='')
    process.wait()
    return part_id

def split_file(input_file, num_parts, output_dir, prefix):
    """把 input_file 拆成 num_parts 个小文件"""
    with open(input_file, "r") as f:
        lines = f.readlines()
    total = len(lines)
    part_size = (total + num_parts - 1) // num_parts

    files = []
    for i in range(num_parts):
        part_lines = lines[i * part_size : (i + 1) * part_size]
        if not part_lines:
            break
        part_file = os.path.join(output_dir, f"{prefix}_{i+1}.scp")
        with open(part_file, "w") as f:
            f.writelines(part_lines)
        files.append(part_file)
    return files

def eval(pred_dir, output_dir, num_jobs=32):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # 准备 pred.scp
    pred_scp = os.path.join(output_dir, "pred.scp")
    with open(pred_scp, "w") as f:
        for wav_path in sorted(glob.glob(os.path.join(pred_dir, "*.wav"))):
            utt_id = os.path.splitext(os.path.basename(wav_path))[0]
            abs_path = os.path.abspath(wav_path)
            f.write(f"{utt_id} {abs_path}\n")

    # 拆分 pred.scp
    pred_parts = split_file(pred_scp, num_jobs, output_dir, "pred_part")

    # 每个子进程任务参数
    tasks = []
    for i, part_file in enumerate(pred_parts, start=1):
        output_file = os.path.join(output_dir, f"result_part_{i}.jsonl")
        tasks.append((part_file, output_file, i))

    # 多进程跑
    with Pool(processes=num_jobs) as pool:
        for i, part_id in enumerate(pool.imap_unordered(run_scorer, tasks), start=1):
            print(f"✅ Progress: {i}/{len(tasks)} parts finished (last done: Part {part_id})")

    # ✅ 合并所有 result_part_*.jsonl
    merged_file = os.path.join(pred_dir, "merge", "dnsmos.jsonl")
    os.makedirs(os.path.dirname(merged_file), exist_ok=True)
    with open(merged_file, "w", encoding="utf-8") as fout:
        for part_file in sorted(glob.glob(os.path.join(output_dir, "result_part_*.jsonl"))):
            with open(part_file, "r", encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
    print(f"🎉 All results merged into {merged_file}")

    print(f"\n🎉 All done in {(time.time() - start_time)/60:.2f} minutes.")

def get_average_score(root_dir): 
    records = []

    # 遍历所有 .jsonl 文件
    for filename in os.listdir(root_dir):
        if filename.endswith('.jsonl'):
            filepath = os.path.join(root_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        file_id = data['key'].replace('.wav', '')
                        if all(k in data for k in ['dns_overall', 'dns_p808']):
                            records.append({
                                '文件ID': file_id,
                                'dns_overall': data['dns_overall'],
                                'dns_p808': data['dns_p808']
                            })
                    except json.JSONDecodeError:
                        print(f"⚠️ 跳过无效行：{line}")

    df = pd.DataFrame(records)
    if not df.empty:
        print(f"✅ 成功数: {len(df)}")
        print(f"📊 各指标平均值：")
        print(f"平均 DNSMOS Overall: {df['dns_overall'].mean():.2f}")
        print(f"平均 DNSMOS P808: {df['dns_p808'].mean():.2f}")
    else:
        print("⚠️ 没有有效的评分记录。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=str, required=True, help="预测音频所在路径")
    parser.add_argument("--no_eval", action='store_true', help="不评估，只计算平均分")
    args = parser.parse_args()

    pred_dir = args.pred_dir
    output_dir = os.path.join(pred_dir, "eval_dnsmos")
    os.makedirs(output_dir, exist_ok=True)

    if not args.no_eval:
        eval(pred_dir, output_dir, num_jobs=20)
        
    get_average_score(output_dir)

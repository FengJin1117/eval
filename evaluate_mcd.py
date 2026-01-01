# evaluate_mcd.py
import warnings
warnings.filterwarnings("ignore")

import os
import json
import glob
import argparse
import subprocess
import time
from multiprocessing import Pool
import pandas as pd

# =========================
# Utils
# =========================

def build_scp(wav_dir, scp_path):
    with open(scp_path, "w") as f:
        for wav in sorted(glob.glob(os.path.join(wav_dir, "*.wav"))):
            utt = os.path.splitext(os.path.basename(wav))[0]
            f.write(f"{utt} {os.path.abspath(wav)}\n")


def split_file(input_file, num_parts, output_dir, prefix):
    with open(input_file) as f:
        lines = f.readlines()

    part_size = (len(lines) + num_parts - 1) // num_parts
    part_files = []

    for i in range(num_parts):
        part_lines = lines[i * part_size:(i + 1) * part_size]
        if not part_lines:
            break
        part_path = os.path.join(output_dir, f"{prefix}_{i+1}.scp")
        with open(part_path, "w") as f:
            f.writelines(part_lines)
        part_files.append(part_path)

    return part_files


def run_scorer(args):
    gt_scp, pred_scp, output_jsonl, part_id = args
    cmd = [
        "PYTHONWARNINGS=ignore"
        "python", "../versa/versa/bin/scorer.py",
        "--score_config", "../versa/egs/separate_metrics/mcd_f0.yaml",
        "--gt", gt_scp,
        "--pred", pred_scp,
        "--output_file", output_jsonl,
        "--io", "soundfile"
    ]
    subprocess.run(cmd, check=True)
    return part_id


# =========================
# Core API
# =========================

def evaluate_mcd(
    gt_dir: str,
    pred_dir: str,
    output_dir: str = None,
    num_jobs: int = 24,
):
    """
    Evaluate MCD/F0 metrics.

    Returns:
        dict: { "mcd": float, "f0rmse": float, "f0corr": float }
    """
    start = time.time()

    if output_dir is None:
        output_dir = os.path.join(pred_dir, "eval_mcd_f0")
    os.makedirs(output_dir, exist_ok=True)

    gt_scp = os.path.join(output_dir, "gt.scp")
    pred_scp = os.path.join(output_dir, "pred.scp")

    build_scp(gt_dir, gt_scp)
    build_scp(pred_dir, pred_scp)

    pred_parts = split_file(pred_scp, num_jobs, output_dir, "pred_part")

    tasks = []
    for i, part in enumerate(pred_parts, start=1):
        out_jsonl = os.path.join(output_dir, f"result_part_{i}.jsonl")
        tasks.append((gt_scp, part, out_jsonl, i))

    with Pool(processes=num_jobs) as pool:
        list(pool.imap_unordered(run_scorer, tasks))

    metrics = collect_average_metrics(output_dir)

    print(f"🎉 Done in {time.time() - start:.2f}s | {metrics}")
    return metrics


def collect_average_metrics(eval_dir):
    records = []

    for fn in os.listdir(eval_dir):
        if fn.endswith(".jsonl"):
            with open(os.path.join(eval_dir, fn)) as f:
                for line in f:
                    data = json.loads(line)
                    if all(k in data for k in ["mcd", "f0rmse", "f0corr"]):
                        records.append({
                            "mcd": data["mcd"],
                            "f0rmse": data["f0rmse"],
                            "f0corr": data["f0corr"],
                        })

    df = pd.DataFrame(records)
    if df.empty:
        return {"mcd": 0.0, "f0rmse": 0.0, "f0corr": 0.0}

    return {
        "mcd": float(df["mcd"].mean()),
        "f0rmse": float(df["f0rmse"].mean()),
        "f0corr": float(df["f0corr"].mean()),
    }


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--num_jobs", type=int, default=24)
    args = parser.parse_args()

    evaluate_mcd(
        gt_dir=args.gt_dir,
        pred_dir=args.pred_dir,
        num_jobs=args.num_jobs,
    )

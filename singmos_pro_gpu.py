import os
import json
import torch
import librosa
import multiprocessing as mp
from tqdm import tqdm
import csv

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------
# 加载模型（进程内执行）
# ---------------------------------------------------------
def load_model_for_gpu(predictor_name: str, gpu_id: int):
    torch.cuda.set_device(gpu_id)
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v1.1.1",
        predictor_name,
        trust_repo=True,
        map_location=f"cuda:{gpu_id}"
    )
    predictor.to(f"cuda:{gpu_id}")
    predictor.eval()
    return predictor

from pydub import AudioSegment

def get_wav_duration(wav_path):
    """
    读取 .wav 文件并返回时长（秒单位）。
    
    参数:
        wav_path (str): .wav 文件的路径。
    
    返回:
        float: 音频时长（秒）。
    """
    try:
        audio = AudioSegment.from_wav(wav_path)
        return len(audio) / 1000.0  # 毫秒转秒
    except Exception as e:
        print(f"❌ 无法读取文件 {wav_path}: {e}")
        return None

# ---------------------------------------------------------
# 单文件夹评测逻辑（不负责多GPU调度）
# ---------------------------------------------------------
def batch_singmos(folder, predictor, gpu_id: int, predictor_name: str = "singmos_pro", output_jsonl: bool = True):
    results = {}
    wav_files = [f for f in os.listdir(folder) if f.lower().endswith(".wav")]

    jsonl_path = os.path.join(folder, "eval")
    if output_jsonl:
        os.makedirs(jsonl_path, exist_ok=True)
        jsonl_file = os.path.join(jsonl_path, f"{predictor_name}.jsonl")
        f_jsonl = open(jsonl_file, "w", encoding="utf8")

    for fname in tqdm(wav_files, desc=f"[GPU{gpu_id}] {os.path.basename(folder)}", leave=False):
        wav_path = os.path.join(folder, fname)

        if get_wav_duration(wav_path) < 0.2:
            continue  # 跳过过短的音频

        # Load wav
        # wave, sr = librosa.load(path, sr=None, mono=True)
        wave, sr = librosa.load(wav_path, sr=16000, mono=True) # 按照16hz读入
        wave = torch.tensor(wave).unsqueeze(0).to(f"cuda:{gpu_id}")
        length = torch.tensor([wave.shape[1]]).to(f"cuda:{gpu_id}")

        # with torch.no_grad():
        #     score = predictor(wave, length)

        # ★★★★★ 在这里捕获错误 ★★★★★
        try:
            score = predictor(wave, length)
        except Exception as e:
            print(f"[GPU{gpu_id}] ❌ wav 文件出错: {wav_path}")
            print(f"错误类型: {type(e)}")
            print(f"错误信息: {e}")
            raise e  # 继续让 multiprocessing 感知错误，但已经打印了文件名
        # ★★★★★ ★★★★★ ★★★★★ ★★★★★

        score_value = float(round(score.item(), 2))
        results[fname] = score_value

        # 写入 JSONL
        if output_jsonl:
            f_jsonl.write(json.dumps({
                "filename": fname,
                predictor_name: score_value
            }) + "\n")

    if output_jsonl:
        f_jsonl.close()

    # 平均分
    avg_score = sum(results.values()) / len(results) if len(results) > 0 else 0.0
    return results, avg_score


# ---------------------------------------------------------
# 多 GPU worker（每个进程跑一个文件夹）
# ---------------------------------------------------------
def worker_process(args):
    folder, predictor_name, gpu_id = args
    predictor = load_model_for_gpu(predictor_name, gpu_id)
    results, avg = batch_singmos(folder, predictor, gpu_id, predictor_name=predictor_name, output_jsonl=True)

    # 清理 GPU 显存
    del predictor
    torch.cuda.empty_cache()

    return folder, avg


# ---------------------------------------------------------
# 调度器：把多个 folder 分配到多个 GPU，多余的进入队列等待
# ---------------------------------------------------------
def dispatch_jobs_to_gpus(wav_dirs, available_gpus, predictor_name="singmos_pro"):
    jobs = []
    for i, folder in enumerate(wav_dirs):
        gpu_id = available_gpus[i % len(available_gpus)]
        jobs.append((folder, predictor_name, gpu_id))

    # 进程池（进程数 = GPU 数量）
    with mp.Pool(processes=len(available_gpus)) as pool:
        results = pool.map(worker_process, jobs)

    return results

def get_field_from_path(path, n=2):
    """
    从路径中取倒数第 n 个字段。
    n=1 表示取最后一个字段，n=2 表示倒数第二个字段，以此类推。
    """
    parts = path.strip("/").split("/")
    return parts[-n]

# ---------------------------------------------------------
# 高层接口：你只需要调这个函数
# ---------------------------------------------------------
def run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, predictor_name="singmos_pro", genre_index=3, output_csv=True):
    results =  dispatch_jobs_to_gpus(wav_dirs, available_gpus, predictor_name)

    # 输出平均分
    # for folder, avg in results:
    #     print(f"{folder}: {avg:.2f}")

    new_dict = {}

    for folder, avg in results:
        # 取倒数第二个字段 -> genre
        # genre = os.path.basename(os.path.dirname(folder))
        genre = get_field_from_path(folder, n=genre_index)
        new_dict[genre] = avg
        print(f"{genre}: {avg:.2f}")

    # 输出 CSV
    if output_csv:
        output_dir = f"{predictor_name}_results"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{os.path.basename(root_dir)}_{predictor_name}_16k.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["genre", "singmos_pro"])   # 表头
            for genre, score in new_dict.items():
                writer.writerow([genre, f"{score:.2f}"])
            print(f"✅ 测评结果已保存到 {output_path}")


def run_opencpop(available_gpus):
    # opencpop
    # model_name = "stylesinger"
    # model_name = "techsinger"
    model_name = "tcsinger"
    root_dir = f"/data7/fwh/benchmark/suno_svs_infer/opencpop_{model_name}"
    wav_dirs = [
        root_dir
    ]
    for predictor_name in ["singmos_v1", "singmos_pro"]:
        print(f"Running evaluation: {predictor_name}")
        run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, predictor_name=predictor_name, genre_index=1, output_csv=False)

# ---------------------------------------------------------
# 示例运行
# ---------------------------------------------------------
if __name__ == "__main__":
    # print("Starting SingMOS Pro multi-GPU evaluation...")
    # 你想使用的 GPU
    available_gpus = [1, 2, 3, 4, 6, 7]

    # run_opencpop(available_gpus)

    # Suno GTSinger系模型测评
    # model = "diffsinger"
    # model = "stylesinger"
    # model = "techsinger"
    # model = "tcsinger"
    # root_dir = f"/data7/fwh/benchmark/suno_svs_infer/suno_{model}"
    # # 遍历所有需要评测的 wav 文件夹（每个 genre 一个）
    # wav_dirs = [
    #     os.path.join(root_dir, genre)
    #     for genre in os.listdir(root_dir)
    # ]
    # for predictor_name in ["singmos_v1", "singmos_pro"]:
    #     print(f"Running evaluation: {predictor_name}")
    #     run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, predictor_name=predictor_name, genre_index=1)

    # suno for mos
    root_dir = "/data7/fwh/benchmark/suno_mos"
    # 遍历所有需要评测的 wav 文件夹（每个 genre 一个）
    wav_dirs = [
        os.path.join(root_dir, genre)
        for genre in os.listdir(root_dir)
    ]
    for predictor_name in ["singmos_v1", "singmos_pro"]:
        print(f"Running evaluation: {predictor_name}")
        run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, predictor_name=predictor_name, genre_index=1)

    # root_dir = "/data7/fwh/benchdata/suno_score"
    # # 遍历所有需要评测的 wav 文件夹（每个 genre 一个）
    # wav_dirs = [
    #     os.path.join(root_dir, genre, "wavs")
    #     for genre in os.listdir(root_dir)
    # ]
    # run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, genre_index=2)

    # opencpop
    # wav_dirs = [
    #     "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_naive_rnn_dp/exp/svs_train_naive_rnn_dp_raw_phn_None_zh/opencpop_stars/test/wav",
    #     "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_xiaoice/exp/svs_train_xiaoice_raw_phn_None_zh/opencpop_stars/test/wav",
    #     "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_visinger/exp/svs_visinger_normal/opencpop_stars/test/wav",
    #     "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_visinger2/exp/svs_visinger2_normal/opencpop_stars/test/wav"
    # ]

    # run_eval_multi_gpu(root_dir, wav_dirs, available_gpus, genre_index=4)

    # root_dir = "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_visinger/exp/svs_visinger_normal"
    # root_dirs = [
    #     "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_naive_rnn_dp/exp/svs_train_naive_rnn_dp_raw_phn_None_zh",
        # "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_xiaoice/exp/svs_train_xiaoice_raw_phn_None_zh",
        # "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_visinger/exp/svs_visinger_normal",
        # "/data7/fwh/espnet/egs2/opencpop_benchmark/svs1/checkpoints/opencpop_visinger2/exp/svs_visinger2_normal"
    # ]
    
    
    # for root_dir in root_dirs:
    #     print(f"评测模型：", os.path.basename(root_dir))
    #     # 遍历所有需要评测的 wav 文件夹（每个 genre 一个）
    #     wav_dirs = [
    #         os.path.join(root_dir, genre, "test", "wav")
    #         for genre in os.listdir(root_dir) 
    #         if "suno" in genre.lower() 
    #         # if "suno" in genre.lower() and "pop" not in genre.lower()
    #     ]
    #     # print(wav_dirs)

    #     run_eval_multi_gpu(root_dir, wav_dirs, available_gpus)

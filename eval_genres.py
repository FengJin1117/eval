# eval_genres.py
import os
import csv
from evaluate_mcd import evaluate_mcd

def resolve_gt_dir(pred_genre_dir):
    genre = os.path.basename(pred_genre_dir)
    return f"/data7/fwh/benchdata/suno_score_fix/{genre}/wavs"


def eval_genres(root_dir, output_dir, num_jobs=24):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(
        output_dir,
        f"{os.path.basename(root_dir)}_mcd_f0_16k.csv"
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["genre", "mcd", "f0rmse", "f0corr"])

        for genre in sorted(os.listdir(root_dir)):
            pred_dir = os.path.join(root_dir, genre)
            if not os.path.isdir(pred_dir):
                continue

            gt_dir = resolve_gt_dir(pred_dir)
            print(f"\n🎵 Evaluating {genre}")

            metrics = evaluate_mcd(
                gt_dir=gt_dir,
                pred_dir=pred_dir,
                num_jobs=num_jobs,
            )

            writer.writerow([
                genre,
                f"{metrics['mcd']:.3f}",
                f"{metrics['f0rmse']:.3f}",
                f"{metrics['f0corr']:.4f}",
            ])
            f.flush()

    print(f"\n✅ Saved to {csv_path}")


if __name__ == "__main__":
    root_dir = "/data7/fwh/benchmark/fix_svs_infer/suno_diffsinger"
    output_dir = os.path.dirname(root_dir)
    eval_genres(root_dir, output_dir)

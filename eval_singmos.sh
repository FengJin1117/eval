pred_dirs=(
    # "/data7/fwh/benchmark/suno_svs_infer/opencpop_diffsinger"
    "/data7/fwh/benchmark/suno_svs_infer/opencpop_stylesinger"
)

for PRED_DIR in "${pred_dirs[@]}"; do
    CUDA_VISIBLE_DEVICES=5 python eval_singmos.py --pred_dir "$PRED_DIR"
    # python eval_singmos.py --pred_dir "$PRED_DIR" --no_eval  # 只算平均分
    echo "预测数据集: $PRED_DIR"
done
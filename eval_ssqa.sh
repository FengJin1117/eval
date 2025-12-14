pred_dirs=(
#   "/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h/decode_vits_200epoch/eval/wav"
  # "/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h_filter/decode_vits_200epoch/eval/wav"
  # "/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h_same_token/decode_vits_250epoch/eval/wav"
  # "/data2/fwh/espnet/egs2/opencpop/svs1/exp/svs_visinger2_opencpop_batch4/decode_vits_200epoch/test/wav"
  # "/data2/fwh/espnet/egs2/acesinger_infer/svs1/exp/svs_visinger2_5h/decode_vits_200epoch/eval/wav"
  # "/data2/fwh/espnet/egs2/opencpop_infer/svs1/exp/svs_visinger2_opencpop_batch4/decode_vits_200epoch/test/wav"
  "/data7/fwh/syn_10k_0907_cut_wavs"
  # "/data7/fwh/data/opencpop/segments/testset"
)

for PRED_DIR in "${pred_dirs[@]}"; do
    CUDA_VISIBLE_DEVICES=2 python eval_ssqa.py --pred_dir "$PRED_DIR"
    # python eval_ssqa.py --pred_dir "$PRED_DIR" --no_eval  # 只算平均分
    echo "预测数据集: $PRED_DIR"
done
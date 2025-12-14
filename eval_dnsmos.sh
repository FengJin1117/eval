# # 预测
# PRED_DIR="/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h/decode_vits_200epoch/eval/wav"

# # conda activate versa
# python eval_dnsmos.py \
#     --pred_dir "$PRED_DIR" \
#     # --no_eval # 只计算平均分，不跑versa评估
# echo "预测数据集: $PRED_DIR"

# PRED_DIR="/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h_filter/decode_vits_200epoch/eval/wav"
# python eval_dnsmos.py \
#     --pred_dir "$PRED_DIR" \
#     # --no_eval # 只计算平均分，不跑versa评估
# echo "预测数据集: $PRED_DIR"

# PRED_DIR="/data2/fwh/espnet/egs2/acesinger_infer_yue_300_filter_human/svs1/exp/svs_visinger2_25h_same_token/decode_vits_250epoch/eval/wav"
# python eval_dnsmos.py \
#     --pred_dir "$PRED_DIR" \
#     # --no_eval # 只计算平均分，不跑versa评估
# echo "预测数据集: $PRED_DIR"


PRED_DIR="/data7/fwh/syn_10k_0907_cut_wavs"
python eval_dnsmos.py \
    --pred_dir "$PRED_DIR" \
    # --no_eval # 只计算平均分，不跑versa评估
echo "预测数据集: $PRED_DIR"
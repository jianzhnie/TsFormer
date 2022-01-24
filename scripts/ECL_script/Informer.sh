export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-24 10:12:01
 # @LastEditTime: 2022-01-24 10:56:46
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL_96_96 \
        --model informer \
        --data custom \
        --features S \
        --seq_len 96 \
        --label_len 48 \
        --pred_len 96 \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 1 \
        --dec_in 1 \
        --c_out 1 \
        --des 'Exp' \
        --itr 1

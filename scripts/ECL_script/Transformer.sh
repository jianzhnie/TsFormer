export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-24 10:12:01
 # @LastEditTime: 2022-01-25 10:21:24
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model transformer \
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
        --embed fixed \
        --des 'Exp' \
        --itr 1


python -u run_autoformer.py \
        --is_training 1 \
        --root_path ./data/electricity/ \
        --data_path electricity.csv \
        --model_id ECL \
        --model transformer \
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
        --embed timeF \
        --des 'Exp' \
        --itr 1

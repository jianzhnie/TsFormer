export CUDA_VISIBLE_DEVICES=1
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-13 15:25:34
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_24 \
  --model rnn \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 24 \
  --input_size 2 \
  --hidden_size 256 \
  --num_layers 1 \
  --output_size 48 \
  --des 'Exp' \
  --itr 1

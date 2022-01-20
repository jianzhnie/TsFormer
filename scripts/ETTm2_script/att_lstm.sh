export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-20 17:04:16
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2 \
  --model attlstm \
  --data RNNData \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --freq t \
  --input_size 7 \
  --hidden_size 256 \
  --num_layers 1 \
  --output_size 96 \
  --train_epochs 20 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --gpu 0 \
  --des 'Exp' \
  --itr 1

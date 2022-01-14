export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-14 15:50:12
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run.py \
  --is_training 1 \
  --root_path ./data/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1 \
  --model lstm \
  --data RNNData \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --input_size 7 \
  --hidden_size 256 \
  --num_layers 1 \
  --output_size 96 \
  --train_epochs 20 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --gpu 0 \
  --des 'Exp' \
  --itr 1

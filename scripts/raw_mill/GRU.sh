export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-14 18:15:21
 # @LastEditors: jianzhnie
 # @Description:
 #
###

python -u run.py \
  --is_training 1 \
  --root_path ./data/raw_milla/ \
  --data_path data1_process.csv \
  --model_id raw_milla_data1 \
  --model gru \
  --data RNNData \
  --target target1 \
  --features S \
  --seq_len 1 \
  --label_len 1 \
  --pred_len 1 \
  --input_size 1 \
  --hidden_size 256 \
  --num_layers 1 \
  --output_size 1 \
  --train_epochs 20 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --gpu 0 \
  --des 'Exp' \
  --itr 1

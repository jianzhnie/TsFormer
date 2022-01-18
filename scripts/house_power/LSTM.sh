export CUDA_VISIBLE_DEVICES=0
###
 # @Author: jianzhnie
 # @Date: 2022-01-13 15:20:03
 # @LastEditTime: 2022-01-18 14:42:06
 # @LastEditors: jianzhnie
 # @Description:
 #
###
python -u run.py \
  --is_training 1 \
  --root_path ./data/house_power/ \
  --data_path UCI_household_power_consumption_synth_hour.csv \
  --model_id UCI_household_power \
  --model lstm \
  --data RNNData \
  --target Global_active_power \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --input_size 1 \
  --hidden_size 256 \
  --num_layers 1 \
  --output_size 96 \
  --train_epochs 20 \
  --batch_size  32 \
  --learning_rate 0.0005 \
  --gpu 0 \
  --des 'Exp' \
  --itr 1

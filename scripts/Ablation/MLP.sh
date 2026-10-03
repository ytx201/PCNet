#!/bin/bash

# MLP (Multi-Layer Perceptron) Model Experiments
# Using PCNet with pcnet_mode=MLP

# ETTh1 Experiment
model_name=PCNet
pcnet_mode=MLP

root_path_name=./dataset/
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running ETTh1 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --template_num 8 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

# ETTh2 Experiment
root_path_name=./dataset/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running ETTh2 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --template_num 6 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

# ETTm1 Experiment
root_path_name=./dataset/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running ETTm1 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --template_num 52 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

# ETTm2 Experiment
root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running ETTm2 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 96 \
      --template_num 33 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done

# Electricity Experiment
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running Electricity experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --cycle 168 \
      --template_num 416 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

# Solar Experiment
root_path_name=./dataset/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running Solar experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --cycle 144 \
      --template_num 248 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --use_revin 0 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.003 --random_seed $random_seed
done
done

# Traffic Experiment
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running Traffic experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 862 \
      --cycle 168 \
      --template_num 1 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 16 --learning_rate 0.003 --random_seed $random_seed
done
done

# Weather Experiment
root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    echo "Running Weather experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --template_num 168 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --patience 5 \
      --dropout 0.5 \
      --itr 1 --batch_size 64 --learning_rate 0.001 --random_seed $random_seed
done
done

# PEMS03 Experiment
root_path_name=./dataset/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    echo "Running PEMS03 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 358 \
      --cycle 288 \
      --template_num 536 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --use_revin 0 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

# PEMS04 Experiment
root_path_name=./dataset/
data_path_name=PEMS04.npz
model_id_name=PEMS04
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    echo "Running PEMS04 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 307 \
      --cycle 288 \
      --template_num 736 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --use_revin 0 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

# PEMS07 Experiment
root_path_name=./dataset/
data_path_name=PEMS07.npz
model_id_name=PEMS07
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    echo "Running PEMS07 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 883 \
      --cycle 288 \
      --template_num 1264 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --use_revin 0 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

# PEMS08 Experiment
root_path_name=./dataset/
data_path_name=PEMS08.npz
model_id_name=PEMS08
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    echo "Running PEMS08 experiment with pred_len=$pred_len, seed=$random_seed"
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 170 \
      --cycle 288 \
      --template_num 52 \
      --pcnet_mode $pcnet_mode \
      --train_epochs 30 \
      --use_revin 1 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

echo "All MLP experiments completed!"
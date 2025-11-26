export CUDA_VISIBLE_DEVICES="0,1,2,3"

model=iTransformer
methods_h='iTransformer'

gpu_loc=0
seq_len=96
pre_lens_h="96 192 336 720"
filename=ETTh

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python run.py \
    --is_training 1 \
    --root_path ../datasets/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id 'ETTh1_'$seq_len'_'$pred_len'_'$method \
    --data ETTh1 \
    --model $model \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --itr 1 \
    --gpu $gpu_loc
done
done

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python run.py \
    --is_training 1 \
    --root_path ../datasets/ETT-small/ \
    --data_path ETTh2.csv \
    --model_id 'ETTh2_'$seq_len'_'$pred_len'_'$method \
    --data ETTh2 \
    --model $model \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --e_layers 2 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 256 \
    --d_ff 256 \
    --itr 1 \
    --gpu $gpu_loc
done
done


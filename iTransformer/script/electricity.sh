export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=96

model=iTransformer
methods_h='iTransformer'

gpu_loc=0
percent=100

pre_lens_h='96 192 336 720'
filename=Electricity

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python run.py \
    --is_training 1 \
    --root_path ../datasets/electricity/ \
    --data_path electricity.csv \
    --model_id 'Electricity_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --model $model \
    --features M \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size 16 \
    --learning_rate 0.0005 \
    --train_epochs 10 \
    --e_layers 3 \
    --enc_in 321 \
    --dec_in 321 \
    --c_out 321 \
    --d_model 512 \
    --d_ff 512 \
    --itr 1 \
    --gpu $gpu_loc
done
done


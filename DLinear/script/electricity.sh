export CUDA_VISIBLE_DEVICES="0,1,2"

seq_len=512

model=DLinear
methods_h='DLinear'

gpu_loc=0
percent=100

pre_lens_h='96 192 336 720'
filename=Electricity

for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
lr=0.0001
bs=32
python main.py \
    --root_path ../datasets/electricity/ \
    --data_path electricity.csv \
    --model_id 'Electricity_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --method $method \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --enc_in 321 \
    --c_out 321 \
    --freq 0 \
    --percent $percent \
    --itr 1 \
    --model $model \
    --kernel_size 25 \
    --cos 1 \
    --tmax 10 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done


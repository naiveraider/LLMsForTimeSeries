export CUDA_VISIBLE_DEVICES="0,1,2"
seq_len=512
percent=100
filename=traffic.txt 

gpu_loc=0
tag_file=main.py
model=DLinear
methods_h='DLinear'
pre_lens_h="96 192 336 720"
lr=0.0001
bs=32
for pred_len in $pre_lens_h;
do
for method in $methods_h;
do
python $tag_file \
    --root_path ../datasets/traffic/ \
    --data_path traffic.csv \
    --model_id 'traffic_'$seq_len'_'$pred_len'_'$method \
    --data custom \
    --method $method \
    --seq_len $seq_len \
    --label_len 0 \
    --pred_len $pred_len \
    --batch_size $bs \
    --learning_rate $lr \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --enc_in 862 \
    --c_out 862 \
    --freq 0 \
    --all 1 \
    --percent $percent \
    --itr 1 \
    --model $model \
    --kernel_size 25 \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --gpu_loc $gpu_loc \
    --save_file_name $filename
done
done


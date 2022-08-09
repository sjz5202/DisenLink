#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')

# nembed_lst=(32 16)
beta_lst=(1)

gpu=2
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=1
m=5


for sub in "${sub_dataset1[@]}"; do
		python ./main_factorgcn.py --dataset $dataset1 \
            --sub_dataset $sub\
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m 
done 

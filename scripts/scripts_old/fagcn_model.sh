#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub_dataset=('Reed98')

# nembed_lst=(32 16)
beta_lst=(1)
#dataset=('squirrel' 'crocodile')
dataset=("fb100")
gpu=0
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=3
m=5


for data in "${dataset[@]}"; do
		python ./main_fagcn.py --dataset $data \
            --sub_dataset $sub_dataset\
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m 
done 

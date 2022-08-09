#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
#sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset1=("TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
mini=(0)
# nembed_lst=(32 16)
beta_lst=(1)

gpu=4
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=10
m=5
dataset='wisconsin'

for mi in "${mini[@]}"; do
		python ./main_disgcn.py --dataset $dataset \
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --miniid $mi \
            --m $m 
done 
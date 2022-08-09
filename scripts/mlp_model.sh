#!/bin/bash
dataset1="twitch-e"
dataset3="year"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
mini=(0)
# nembed_lst=(32 16)
beta_lst=(1)
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=10
m=5
ind=(0)
id=1
nhidden=(512)
lr=0.001
dataset='wisconsin'
sub_dataset2=('JohnsHopkins55')
sub_dataset='Amherst41'
gpu=1
id_year=(0)
for hid in "${nhidden[@]}"; do
    for id in "${id_year[@]}"; do
		  python baseline/main_mlp.py --dataset $dataset \
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --nhidden $hid \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m \
            --sub_dataset $sub_dataset\
            --miniid $id
    done
done 
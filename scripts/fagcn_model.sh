#!/bin/bash
dataset1="twitch-e"
dataset2="wisconsin"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub='JohnsHopkins55'
# nembed_lst=(32 16)
beta_lst=(1)
dataset='year'
gpu=1
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=10
m=5
mi=1
mini=(0)
nhidden=(512)
for hid in "${nhidden[@]}"; do
		python ./main_fagcn.py --dataset $dataset2 \
            --sub_dataset $sub\
            --lr $lr \
            --nhidden $hid \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m  \
            --miniid $mi
done 


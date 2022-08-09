#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub="Reed98"

# nembed_lst=(32 16)
beta_lst=(1)

gpu=0
lr=0.001
weight_decay=5e-4
nembed=32
epochs=2000
run=5
m=5
hid=(128 256)

for hi in "${hid[@]}"; do
		python ./main_vgae.py --dataset $dataset2 \
            --sub_dataset $sub\
            --lr $lr \
            --nhidden $hi\
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m 
done 
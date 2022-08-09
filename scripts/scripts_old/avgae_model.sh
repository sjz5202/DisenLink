#!/bin/bash
dataset1="twitch-e"
dataset2="photo"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub='Reed98'
# nembed_lst=(32 16)
beta_lst=(1)

gpu=4
lr=0.001
nfactor=3
weight_decay=5e-4
nfeat=128
nhidden=(128)
nembed=32
epochs=2000
layer=1
temperature=1
loss_weight=20
run=10
m=5



for hi in "${nhidden[@]}"; do
		python ./main_avgae.py --dataset $dataset2 \
            --sub_dataset $sub\
            --nhidden $nhidden \
            --lr $lr \
            --nfactor $nfactor \
            --weight_decay $weight_decay \
            --nfeat $nfeat \
            --nhidden $hi \
            --nembed $nembed \
            --epochs $epochs \
            --layer $layer \
            --temperature $temperature \
            --loss_weight $loss_weight \
            --run $run    \
            --m $m \
            --gpu $gpu   
done 
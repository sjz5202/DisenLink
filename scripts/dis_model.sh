#!/bin/bash
dataset1="squirrel"
dataset2="fb100"
dataset3="twitch-e"
#sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset1=("ENGB" "FR")
#sub_dataset1=("ES" "FR" "RU" "PTBR" "TW")
sub_dataset2=('JohnsHopkins55')
#sub_dataset2=('Amherst41')
#sub_dataset="Johns Hopkins55"
sub_dataset="ENGB"
gpu=7
lr=0.0001
weight_decay=5e-4
nfeat=128
nhidden=256
nembed=32
epochs=2000
layer=1
temperature=1
loss_weight=20
run=1
m=5
#beta_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
#beta_list=(0.7)
#beta_list=(0.5)
#beta_list=(0.85 0.8 0.7 0.5 0.6 0.6)
nfactorlist=(10)
factor=(1 2 3 4)
beta=0.5
for nfactor in "${nfactorlist[@]}"; do
            python ./main_disentangled.py --dataset $dataset3 \
            --sub_dataset $sub_dataset\
            --nhidden $nhidden \
            --lr $lr \
            --beta $beta \
            --nfactor $nfactor \
            --weight_decay $weight_decay \
            --nfeat $nfeat \
            --nembed $nembed \
            --epochs $epochs \
            --layer $layer \
            --temperature $temperature \
            --loss_weight $loss_weight \
            --run $run \
            --gpu $gpu \
            --m $m
done
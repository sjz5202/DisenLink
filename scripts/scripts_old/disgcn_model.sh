#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
#sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
sub_dataset1=("TW")
sub_dataset2=('Amherst41' 'Johns Hopkins55')

# nembed_lst=(32 32 64 64)
beta_lst=(1)

gpu=0
lr=0.001
weight_decay=5e-4
nembed=32
nhid=(16 32)
epochs=2000
run=1
m=5
dataset=('fb100' 'fb100')
subdata=('Amherst41' 'Amherst41')
ind=(0 1)

for id in "${ind[@]}"; do
		python ./main_disgcn.py --dataset ${dataset[id]} \
            --sub_dataset ${subdata[id]}\
            --nhidden ${nhid[id]} \
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m 
done 

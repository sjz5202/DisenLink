#!/bin/bash
dataset1="twitch-e"
dataset2="fb100"
sub_dataset1=("ENGB" "ES" "FR" "PTBR" "RU" "TW")
#sub_dataset1=("ENGB")
#sub_dataset2=('Amherst41' 'Johns Hopkins55')
sub_dataset2=('Amherst41')

# nembed_lst=(32 16)
beta_lst=(1)

gpu=5
lr=0.01
weight_decay=5e-4
nembed=32
epochs=2000
run=5
m=5
#ind=(0 1 2 3 4 5 6 7 8)
#alpha_list=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
ind=(0)
alpha_list=(0.4)
dataset=('citeseer')
for id in "${ind[@]}"; do
        python ./main_gprgnn.py --dataset $dataset \
            --sub_dataset $sub_dataset1\
            --lr $lr \
            --weight_decay $weight_decay \
            --nembed $nembed \
            --epochs $epochs \
            --run $run    \
            --gpu $gpu  \
            --m $m \
            --alpha ${alpha_list[id]}
done 

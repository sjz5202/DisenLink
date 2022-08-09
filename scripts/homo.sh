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
dataset=('chameleon' 'squirrel' 'crocodile' 'texas' 'wisconsin' 'twitch-e' 'twitch-e' 'fb100' 'fb100' 'cora' 'citeseer' 'photo')
sub_dataset2=('as' 'as' 'as' 'as' 'as' 'DE' 'ENGB' 'Amherst41' 'JohnsHopkins55' 'as' 'as' 'as')
ID=(0 1 2 3 4 5 6 7 8 9 10 11)
sub_dataset='Amherst41'
gpu=1
id=1
id_year=(0)
for i in "${ID[@]}"; do
    #for id in "${id_year[@]}"; do
		  python baseline/feature_homophily.py --dataset ${dataset[i]}\
            --sub_dataset ${sub_dataset2[i]}\
            --miniid $id
    #done
done 
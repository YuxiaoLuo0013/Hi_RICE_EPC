#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi
#step=200000
num_envs=8
num_test=3
#learning_rate=1e-4
#num_steps=4
#python -u train_vanilla_PPO.py \
#--learning-rate $learning_rate \
#--num-envs $num_envs \
#--total-timesteps $step >logs/region_ppo/region_ppo.log
step=60000
p_trans=0.07
learning_rate=1e-4
alpha=2.5
wta=1
wtp=1
gpu=0
python -u train.py \
--num-envs $num_envs \
--num-test $num_test \
--learning-rate $learning_rate \
--gpu $gpu \
--p-trans $p_trans \
--alpha $alpha \
--wta $wta \
--wtp $wtp \
--total-timesteps $step >logs/no_contact_risk_{$p_trans}_alpha_{$alpha}_wta_{$wta}_wtp_{$wtp}.log

#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/GBM" ]; then
    mkdir ./logs/GBM
fi

#step=200000
num_envs=1
#learning_rate=1e-4
#num_steps=4
#python -u train_vanilla_PPO.py \
#--learning-rate $learning_rate \
#--num-envs $num_envs \
#--total-timesteps $step >logs/region_ppo/region_ppo.log
step=20000
learning_rate=1e-4
p_trans=0.24
python -u train_GBM.py >logs/GBM/GBM.log

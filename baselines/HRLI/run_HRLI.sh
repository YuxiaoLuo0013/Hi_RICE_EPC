#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/HRLI" ]; then
    mkdir ./logs/HRLI
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
python -u train_HRLI.py \
--num-envs $num_envs \
--learning-rate $learning_rate \
--p-trans $p_trans \
--total-timesteps $step >logs/HRLI/HRLI_024.log

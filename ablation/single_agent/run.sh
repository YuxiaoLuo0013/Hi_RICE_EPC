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
step=100000
p_trans=0.07
learning_rate=1e-4
alpha=2.5
gpu=0
python -u train_single.py \
--num-envs $num_envs \
--num-test $num_test \
--learning-rate $learning_rate \
--gpu $gpu \
--p-trans $p_trans \

--total-timesteps $step >logs/single_agent_p_{$p_trans}.log

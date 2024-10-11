# -*- coding = utf-8 -*-
# @time:2023/11/23 16:34
# Author:Yuxiao
# @File:train_vanilla_PPO.py
import argparse
import os
import random
import time
from distutils.util import strtobool
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers import NormalizeObservation
from torch.utils.tensorboard import SummaryWriter
from gym.envs.registration import register
import warnings
from normalization import RewardScaling, Normalization
from agent import Agent, Agent_lstm,Agent_gcnlstm,c_Agent_lstm

warnings.filterwarnings("ignore")
register(
    id = 'EpidemicModel-v0',  # 使用一个唯一的ID
    entry_point = 'env.env_ma:EpidemicModel',  # 替换为您的环境类路径

)


def make_env(gym_id, test, ptrans,seed):
    def thunk():
        env = gym.make(gym_id, if_test = test, seed = seed,ptrans =ptrans,mask_ratio=mask_ratio,  autoreset = False)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs, -10, 10))
        env = gym.wrappers.FrameStack(env, 7)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

if __name__ == "__main__":
    # TRY NOT TO MODIFY: seeding
    seed = 10000
    num_envs = 10
    p_trans=0.07
    mask_ratio=0
    use_gpu = True
    n_pop=1618304
    n_days=150
    random.seed(seed)
    np.random.seed(seed)
    mean = np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1720642274_0.07_2.5_wta_1.0_wtp_1.0_mean.npy')
    variance = np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1720642274_0.07_2.5_wta_1.0_wtp_1.0_std.npy')
    device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env('EpidemicModel-v0', True,p_trans,mask_ratio,seed + int(i + num_envs)) for i in range(num_envs)]
    )


    print("Sucessfully initialized", num_envs, "environments")

    agent = Agent_lstm(envs, hidden_size = 512).to(device)
    agent.test = True
    agent.load_state_dict(
        torch.load('/EPC_RF/checkpoint/EpidemicModel-v0__train_vanilla_PPO__1__1720642274_0.07_2.5_wta_1.0_wtp_1.0_best_model.pth',map_location='cuda:0'))
    agent.eval()
    next_obs = envs.reset()[0]
    next_obs = (next_obs - mean) / (variance + 1e-8)
    for update in tqdm(range(0, n_days)):
        next_obs = torch.Tensor(next_obs).to(device)
        action, logprob, _, value = agent(next_obs[:, :, :, :11], next_obs)

        next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
        next_obs = (next_obs - mean) / (variance + 1e-8)
    info_log = {}
    info_log['Total_infect'] = []
    info_log['Total_quarantine'] = []
    info_log['Region_cI'] = []
    info_log['Region_cQ'] = []
    Region_Q_log=np.zeros(656)
    daily_I=np.zeros(n_days)
    daily_Q=np.zeros(n_days)
    for i in range(num_envs):
        info_log['Total_infect'].append(info['final_info'][i]['Total_infect'])
        info_log['Total_quarantine'].append(info['final_info'][i]['Total_quarantine'])
        info_log['Region_cI'].append(info['final_info'][i]['Region_I'])
        info_log['Region_cQ'].append(info['final_info'][i]['Region_Q'])
        daily_I=np.vstack((daily_I,info['final_info'][i]['daily_I']))
        daily_Q=np.vstack((daily_Q,info['final_info'][i]['daily_Q']))
        Region_Q_log=np.vstack((Region_Q_log,info['final_info'][i]['Region_Q_log']))
        print('Total infect :', info['final_info'][i]['Total_infect'])
        print('Total quarantine :', info['final_info'][i]['Total_quarantine'])
    print('Mean total infect :', np.mean(info_log['Total_infect']))
#    np.save(f"./daily_record/{p_trans}_daily_I.npy", daily_I[1:,:])
#    np.save(f"./daily_record/{p_trans}_daily_Q.npy", daily_Q[1:,:])
#    np.save(f"./daily_record/{p_trans}_Region_Q.npy", Region_Q_log[1:,:])



    
    print('Mean total quarantine :', np.mean(info_log['Total_quarantine']))
    print('Mean region I:',np.mean(info_log['Region_cI']))
    print('Mean region Q:',np.mean(info_log['Region_cQ']))
    print('Final score:',np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / n_pop) + np.exp(
                            np.array(info_log['Total_quarantine']) / n_pop)))

                           
                            

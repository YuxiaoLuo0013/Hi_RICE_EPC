
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
from agent import Agent, Agent_lstm,Agent_gcnlstm

warnings.filterwarnings("ignore")
register(
    id = 'EpidemicModel-v0',  # 使用一个唯一的ID
    entry_point = 'env.env_expert:EpidemicModel',  # 替换为您的环境类路径
    # 这里可以添加更多的参数，如max_episode_steps等
)


def make_env(gym_id, test, ptrans,seed):
    def thunk():
        env = gym.make(gym_id, if_test = test, seed = seed,ptrans =ptrans,  autoreset = False)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs, -10, 10))
        env = gym.wrappers.FrameStack(env, 7)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


if __name__ == "__main__":
    # TRY NOT TO MODIFY: seeding
    seed = 2024
    num_envs = 10
    p_trans=0.24
    use_gpu = True
    n_pop=1618304
    n_days=150
    random.seed(seed)
    np.random.seed(seed)
    mean = np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1718282929_0.15_mean.npy')
    variance = np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1718282929_0.15_std.npy')
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env('EpidemicModel-v0', True,p_trans,seed + int(i + num_envs)) for i in range(num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # print sucessfully initialized num_envs env
    # envs = gym.wrappers.NormalizeObservation(envs)
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)

    print("Sucessfully initialized", num_envs, "environments")

#    agent = Agent_lstm(envs, hidden_size = 128).to(device)
#    agent.test = True
#    agent.load_state_dict(
#        torch.load('/EPC_RF/checkpoint/EpidemicModel-v0__train_vanilla_PPO__1__1718282929_0.15_best_model.pth'))
#    agent.eval()
    next_obs = envs.reset()[0]
#    next_obs = (next_obs - mean) / (variance + 1e-8)
    for update in tqdm(range(0, n_days)):
        next_obs = torch.Tensor(next_obs).to(device)
        action=np.zeros((num_envs,10))
#        action, logprob, _, value = agent(next_obs[:, :, :, :11], next_obs)

        next_obs, reward, done, _, info = envs.step(action)
#        next_obs = (next_obs - mean) / (variance + 1e-8)
    
    
    info_log = {}
    info_log['Total_infect'] = []
    info_log['Total_quarantine'] = []
    info_log['Region_score'] = []
    daily_I=np.zeros(n_days)
    daily_Q=np.zeros(n_days)
    for i in range(num_envs):
        info_log['Total_infect'].append(info['final_info'][i]['Total_infect'])
        info_log['Total_quarantine'].append(info['final_info'][i]['Total_quarantine'])
        info_log['Region_score'].append(info['final_info'][i]['Region_score'])
        daily_I=np.vstack((daily_I,info['final_info'][i]['daily_I']))
        daily_Q=np.vstack((daily_Q,info['final_info'][i]['daily_Q']))
        print('Total infect :', info['final_info'][i]['Total_infect'])
        print('Total quarantine :', info['final_info'][i]['Total_quarantine'])
    print('Mean total infect :', np.mean(info_log['Total_infect']))

    print('Mean total quarantine:', np.mean(info_log['Total_quarantine']))
    print('Mean region score:', np.mean(info_log['Region_score']))
    
    # Calculate and print the final score
    final_score = np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / n_pop) + 
                          np.exp(np.array(info_log['Total_quarantine']) / n_pop))
    print('Final score:', final_score)

    # Create directory if it doesn't exist
    daily_record_dir = "./daily_record"
    if not os.path.exists(daily_record_dir):
        os.makedirs(daily_record_dir)
        print(f"Created directory: {daily_record_dir}")
    np.save(f"./daily_record/{p_trans}_daily_I_expert.npy", daily_I[1:,:])
    np.save(f"./daily_record/{p_trans}_daily_Q_expert.npy", daily_Q[1:,:])



    
    print('Mean total quarantine :', np.mean(info_log['Total_quarantine']))
    print('Mean region score:',np.mean(info_log['Region_score']))
    print('Final score:',np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / n_pop) + np.exp(
                            np.array(info_log['Total_quarantine']) / n_pop)))

                           
                            

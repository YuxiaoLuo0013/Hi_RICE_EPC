# -*- coding = utf-8 -*-
# @time:2024/6/18 17:48
# Author:Yuxiao
# @File:train_EITL.py
import numpy as np
import time
import gym
from gym import register

if __name__ == "__main__":
    p_trans=[0.07,0.15,0.24]
    register(
        id = 'EpidemicModel-v0',  # 使用一个唯一的ID
        entry_point = 'env.env_Degree:EpidemicModel',  # 替换为您的环境类路径
        # 这里可以添加更多的参数，如max_episode_steps等
    )
    for p_tran in p_trans:
        envs = gym.make('EpidemicModel-v0',ptrans =p_tran)
        action=0.3
        info_log = {}
        info_log['Total_infect'] = []
        info_log['Total_quarantine'] = []
        for i in range(10):
            obs, info = envs.reset()
            start_time = time.time()
            mean_pid_infect_log = []
            mean_pid_home_infect_log = []

            prob_log = np.array([])
            prob_mean_log = []
            for ii in range(150):
                # print("Days:", ii)

                next_obs, reward, done, truncated, info = envs.step(action)
                obs = next_obs
                # print('infected people:', np.sum(envs.delta_I))
                # print('quarantine people:', np.sum(envs.delta_Q_real))

            print(f'Mean total infect_{p_tran} :', info['Total_infect'])
            print(f'Mean total quarantine_{p_tran} :',info['Total_quarantine'])

            info_log['Total_infect'].append(info['Total_infect'])
            info_log['Total_quarantine'].append(info['Total_quarantine'])
            test_score1 = np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / envs.epc_env.n_agent) + np.exp(
                np.array(info_log['Total_quarantine']) / envs.epc_env.n_agent))

        print(f'Mean total infect_{p_tran} :', np.mean(info_log['Total_infect']))
        print(f'Mean total quarantine_{p_tran} :', np.mean(info_log['Total_quarantine']))
        print(f'Testing score_{p_tran}:', test_score1)

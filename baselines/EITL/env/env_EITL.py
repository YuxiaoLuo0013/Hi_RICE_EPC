# -*- coding = utf-8 -*-
# @time:2023/12/4 17:17
# Author:Yuxiao
# @File:env_ma.py.py
import time
import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
from torch.distributions.normal import Normal
from env.EPC_env.EpidemicSimulation_EITL import EpidemicSimulation
from matplotlib import pyplot as plt
from gym.wrappers import NormalizeObservation
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


def prob_func(x):
    x[x < 0] = 0
    return np.tanh(x)


class EpidemicModel(gym.Env):
    def __init__(self, city_name = "Shenzhen",
                 ptrans = 0.15,
                 uptake_scenario = "Uptake 00",
                 is_npi_gather = False,
                 npi_max_gather = 10,
                 rate_iso_p_work_off = 0,
                 p_mask = 0.8,
                 init_level = -1,
                 th1 = np.inf, th2 = np.inf,
                 p_drug = 0.8,
                 is_samp = True,
                 is_spatial = True,
                 samp_rate = 0.1,
                 duration_sigma = 0.2,
                 is_fast_r0 = True,
                 max_iterday = 150,
                 n_imported_day0 = 100,
                 n_imported_daily = 1,
                 seed = 30,
                 if_test = False):
        super(EpidemicModel, self).__init__()
        self.epc_env = EpidemicSimulation(city_name = city_name,
                                          ptrans = ptrans,
                                          uptake_scenario = uptake_scenario,
                                          is_npi_gather = is_npi_gather,
                                          npi_max_gather = npi_max_gather,
                                          rate_iso_p_work_off = rate_iso_p_work_off,
                                          p_mask = p_mask,
                                          init_level = init_level,
                                          th1 = th1, th2 = th2,
                                          p_drug = p_drug,
                                          is_samp = is_samp,
                                          is_spatial = is_spatial,
                                          samp_rate = samp_rate,
                                          duration_sigma = duration_sigma,
                                          is_fast_r0 = is_fast_r0,
                                          max_iterday = max_iterday,
                                          n_imported_day0 = n_imported_day0,
                                          n_imported_daily = n_imported_daily)
        self.seed = seed
        self.sim_date = 0
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (587, 20))
        # this_compartment, next_compartment, transit_countdown, immunity_type, immunity_days, infected_days, quarantine_type, quarantine_countdown, drug_access
        self.action_space=spaces.Box(low=-np.inf,high=np.inf,shape=(self.epc_env.n_imported_day0,3))
        self.action_space = spaces.MultiDiscrete([5 * 3] * 587)
        self.step_num = max_iterday
        self.action_p = np.array([0, 0.05, 0.25, 0.5, 1])
        self.test = if_test

    def reset(self):
        self.log_Q = []
        self.log_I = []
        obs, prob, self.sim_mat, self.sim_res, region_infected, region_controlled = self.epc_env.reset(self.seed)
        self.I = region_infected
        self.Q = np.sum(region_controlled * np.array([0, 0.2, 0.3, 0.5, 1]), axis = 1)
        self.sim_date = self.epc_env.sim_date
        # creat distionary info
        info = {'daily_infect':np.sum(region_infected), 'daily_quarantine':0}
        if self.test == False:
            self.seed += 1

        self.tab_region_person = self.epc_env.tab_person_rank[['pid', 'ranked_hzone']]

        self.num_region = self.epc_env.region_num

        self.stat_prob = prob[:,1]
        self.prob_infecious = prob[:,0]
        # np.save('/EPC_RF/env/num_region.npy', self.num_region)

        return obs, info

    def get_reward(self, region_infected, region_controlled):
        self.delta_I = region_infected
        self.delta_Q_real = np.sum(region_controlled * np.array([0, 0.2, 0.3, 0.5, 1]), axis = 1)

        self.I = self.I + self.delta_I
        self.Q = self.Q + self.delta_Q_real

        reward=0
        return reward
    def step(self, action):
        threshold = action
        individual_action = np.zeros(self.epc_env.n_agent)
        individual_action[self.prob_infecious> threshold[0]]=3
        prob_confine=(self.stat_prob-threshold[1])/self.stat_prob
        prob_confine[prob_confine<0]=0
        # prob_confine[prob_confine>1]=1
        np.random.rand(self.epc_env.n_agent)
        # random 0,1

        individual_action[np.random.rand(self.epc_env.n_agent)<prob_confine]=1



        self.sim_mat, self.sim_res, obs, prob, region_infected, region_controlled = self.epc_env.step(self.sim_mat,
                                                                                                      self.sim_res,
                                                                                                      individual_action)
        self.stat_prob = prob[:,1]
        self.prob_infecious = prob[:,0]

        done = False
        truncated = False  # 添加这行
        reward = self.get_reward(region_infected, region_controlled)
        sim_date = self.epc_env.sim_date
        if sim_date == self.step_num:
            done = True
        info = {'Total_infect':np.sum(self.I), 'Total_quarantine':np.sum(self.Q)}
        # if done==True:
        #     print('Total infected people:',np.sum(self.I))
        #     print('Total quarantine people:',np.sum(self.Q))
        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass



if __name__ == "__main__":

    register(
        id = 'EpidemicModel-v0',  # 使用一个唯一的ID
        entry_point = 'env_EITL:EpidemicModel',  # 替换为您的环境类路径
        # 这里可以添加更多的参数，如max_episode_steps等
    )
    # envs= gym.vector.AsyncVectorEnv(
    #     [make_env('EpidemicModel-v0',seed = 1) for i in range(1)]
    # )
    envs = gym.make('EpidemicModel-v0')
    # model = Agent().to(torch.device("cuda"))
    # model.load_state_dict(
    #     torch.load('/EPC_RF/checkpoint/EpidemicModel-v0__train_vanilla_PPO__1__1705511451_100800.pth'))
    # model.eval()
    #
    #
    # mean=np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1705511451_mean.npy')
    # variance=np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1705511451_std.npy')

    for i in range(1):
        obs, info = envs.reset()
        # print("########Initial State:", sim_res)
        start_time = time.time()
        # wzone_action = np.zeros(6953)
        # hzone_action = np.zeros(587)
        mean_pid_infect_log = []
        mean_pid_home_infect_log = []
        threshold = [0.001,20]
        prob_log = np.array([])
        prob_mean_log = []
        for i in range(150):
            print("Days:", i)

            next_obs, reward, done, truncated, info = envs.step(action=threshold)
            obs = next_obs
            #     print(done)
            #     print(np.mean(reward))
            print("delta I", np.sum(envs.delta_I))
            print("delta Q", np.sum(envs.delta_Q_real))

        # #     print(env.sim_res['daily_infect'])
        print('Total infected people:', sum(envs.sim_res['daily_infect']))
        print('Total quarantine people:', envs.Q.sum())
        #     if done:
        #         obs=env.reset()
        #     obs=next_obs
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print("Execution time: ", execution_time)

        # res=env.sim_res
        # plt.plot(res["daily_infect"][0:100], color="blue")
        # # plt.plot(np.array(res["levels"]) * max(res["daily_infect"]) / 4, color="orangered")
        # plt.show()
        # #save prob_log as npy
        # prob_log.reshape(-1,1)
        # np.save('prob_log.npy',prob_log)
        # np.save('prob_mean_log.npy',np.array(prob_mean_log))
    # print("Average action:",np.mean(action_num))
    # plt.plot(res["current_icu"])
    # plt.hlines(ES.n_cap_icu * ES.samp_rate, 0, 180)
    # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th1, 0, 180)
    # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th2, 0, 180)
    # # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th3, 0, 180)
    # # plt.hlines(ES.n_cap_icu * ES.samp_rate * ES.th4, 0, 180)
    # plt.plot(np.array(res["levels"]) * max(res["current_icu"]) / 4, color="orangered")
    # smoothed = np.convolve(res["current_icu"], np.ones(7) / 7, mode='same')
    # # plt.ylim(0, 100)
    # plt.plot(smoothed, color="green")
    # plt.show()
    #
    # smoothed = np.convolve(res["daily_infect"], np.ones(7) / 7, mode='same')
    # plt.plot(smoothed, color="blue")
    # plt.plot(np.array(res["levels"]) * max(res["daily_infect"]) / 4, color="orangered")
    # plt.show()
    # ES.epc_env.cal_metrics(sim_res)


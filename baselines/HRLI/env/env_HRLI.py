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
from env.EPC_env.EpidemicSimulation1 import EpidemicSimulation




class EpidemicModel(gym.Env):
    def __init__(self, city_name = "Shenzhen",
                 ptrans = 0.2,
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
        self.observation_space1 = spaces.Box(low = -np.inf, high = np.inf, shape = (587, 2))
        self.observation_space2 = spaces.Box(low = -np.inf, high = np.inf, shape = (self.epc_env.n_agent, 11))
        self.observation_space = spaces.Tuple((self.observation_space1, self.observation_space2))
        # this_compartment, next_compartment, transit_countdown, immunity_type, immunity_days, infected_days, quarantine_type, quarantine_countdown, drug_access
        # self.action_space=spaces.Box(low=-np.inf,high=np.inf,shape=(587,3))
        self.action_space = spaces.MultiDiscrete([4] * self.epc_env.n_agent)
        self.step_num = max_iterday
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

        self.prob = prob
        # np.save('/EPC_RF/env/num_region.npy', self.num_region)

        return obs, info

    def get_reward(self, region_infected, region_controlled):

        self.delta_I = region_infected
        self.delta_Q_real = np.sum(region_controlled * np.array([0, 0.2, 0.3, 0.5, 1]), axis = 1)

        self.I = self.I + self.delta_I
        self.Q = self.Q + self.delta_Q_real


        reward_region = -(np.exp((20 * (self.delta_I)) / self.num_region) +np.exp(self.delta_Q_real / self.num_region))


        reward_individual= -(np.exp((20 * (np.sum(self.delta_I))) / np.sum(self.num_region)) +np.exp(np.sum(self.delta_Q_real) / np.sum(self.num_region)))
        reward=(reward_individual, reward_region)
        return reward
    def step(self, action):
        action_ind=np.zeros(action.shape[0])
        p1=np.exp(-action[:,0])/np.sum(np.exp(-action),axis=1)
        p2=(np.exp(-action[:,0])+np.exp(-action[:,1]))/np.sum(np.exp(-action),axis=1)
        p3=(np.exp(-action[:,0])+np.exp(-action[:,1])+np.exp(-action[:,2]))/np.sum(np.exp(-action),axis=1)

        action_ind[self.prob<p1]=0
        action_ind[(self.prob>=p1)&(self.prob<p2)]=1
        action_ind[(self.prob>=p2)&(self.prob<p3)]=2
        action_ind[self.prob>=p3]=3

        # action_ind = np.zeros(action.shape[0])

        self.sim_mat, self.sim_res, obs, prob, region_infected, region_controlled = self.epc_env.step(self.sim_mat,
                                                                                                      self.sim_res,
                                                                                                      action_ind)

        self.prob= prob

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






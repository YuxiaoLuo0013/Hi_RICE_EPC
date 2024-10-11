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
from env.EPC_env.EpidemicSimulation_single import EpidemicSimulation
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
                 ptrans = 0.24,
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
                 alpha=2.0,
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
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (1, 22))
        # this_compartment, next_compartment, transit_countdown, immunity_type, immunity_days, infected_days, quarantine_type, quarantine_countdown, drug_access
        # self.action_space=spaces.Box(low=-np.inf,high=np.inf,shape=(587,3))
        self.action_space = spaces.MultiDiscrete([5 * 3] * 1)
        self.step_num = max_iterday
        self.action_p = np.array([0, 0.05, 0.25, 0.5, 1])
        self.test = if_test
        self.alpha=alpha

    def reset(self):
        self.log_Q = []
        self.log_I = []
        obs, prob, self.sim_mat, region_infected, region_controlled = self.epc_env.reset(self.seed)
        self.I = region_infected
        self.Q = np.sum(region_controlled * np.array([0, 0.2, 0.3, 0.5, 1]), axis = 1)
        self.sim_date = self.epc_env.sim_date
        # creat distionary info
        info = {'daily_infect':np.sum(region_infected), 'daily_quarantine':0}
        if self.test == False:
            self.seed += 1

        self.tab_region_person = self.epc_env.tab_person_rank[['pid', 'ranked_hzone']]

        self.num_region = self.epc_env.region_num

        self.stat_prob = prob[0]
        self.prob_infecious = prob[1]
        self.tab_region_person['risk'] = self.prob_infecious + self.stat_prob
        self.tab_region_person['risk_infecious'] = self.prob_infecious
        # np.save('/EPC_RF/env/num_region.npy', self.num_region)

        return obs, info

    def get_reward(self, region_infected, region_controlled, individual_action):
        self.delta_I = region_infected
        Q_cost1 = np.array([0, 0.2, 0.3, 0.5])[individual_action.astype(int)]
        self.tab_region_person['Q_cost'] = Q_cost1
        self.delta_Q = self.tab_region_person.groupby('ranked_hzone')['Q_cost'].sum().to_numpy()
        self.delta_Q_real = np.sum(region_controlled * np.array([0, 0.2, 0.3, 0.5, 1]), axis = 1)

        self.I = self.I + self.delta_I
        self.Q = self.Q + self.delta_Q_real
        sim_date = self.epc_env.sim_date

        # Q_cost1=np.array([0,0.2,1,5])[individual_action.astype(int)]
        # self.tab_region_person['Q_cost']=Q_cost1
        # Q_cost= self.tab_region_person.groupby('ranked_hzone')['Q_cost'].sum().to_numpy()
        #
        # AVERAGE_Q=15
        # if sim_date<=AVERAGE_Q:
        #     I_cost=((self.epc_env.average_infected[sim_date]*AVERAGE_Q*np.sum(self.epc_env.region_infected[:sim_date,:],axis=0)))*5
        # else:
        #     I_cost=((self.epc_env.average_infected[sim_date]*AVERAGE_Q*np.sum(self.epc_env.region_infected[sim_date-AVERAGE_Q:sim_date,:],axis=0)))*5
        #
        # I_cost=I_cost+self.delta_I*AVERAGE_Q*5
        # reward = -((I_cost/self.num_region) + (Q_cost / self.num_region))
        # print(np.sum(self.epc_env.region_infected[:sim_date+1,:],axis=0))
        # print(np.sum(self.epc_env.region_infected[sim_date-7:sim_date,:],axis=0))
        # self.deltaQ=np.sum(region_controlled[:,1:]*np.array([0.3,0.5,1]),axis=1)
        reward = -((np.sum(20 * (self.delta_I)) ** self.alpha / np.sum(self.num_region)) + (
                    np.sum(self.delta_Q) / np.sum(self.num_region)))
        # reward = -((((self.delta_I))**2/ self.num_region)  + (self.delta_Q / self.num_region))

        # reward = -(self.delta_I / (num_region / 20)) -(self.delta_Q / num_region)
        # reward = -(np.exp((np.sum(self.delta_I) / (self.epc_env.n_agent / 20)))) - (np.exp((np.sum(self.delta_Q) / (self.epc_env.n_agent/20))))
        # reward = -(np.exp(np.sum(self.delta_I) / (self.epc_env.n_agent/20))+np.exp(np.sum(self.delta_Q) / self.epc_env.n_agent))
        # print(np.sum(self.delta_I))
        # print(np.sum(self.delta_Q))
        # reward = -(np.exp((self.delta_I/(num_region/100))))-(np.exp((self.delta_Q/(num_region))))
        # reward = (np.exp((self.delta_Q / num_region)))
        # reward=-self.delta_I
        # reward=self.delta_Q/self.epc_env.n_agent
        reward = np.array([reward])
        return reward

    def step(self, action):
        p = np.zeros(action.shape)


        p[:, 0] = self.action_p[action[:, 0]]
        p[:, 1] = self.action_p[action[:, 1]] * 0.5
        p[:, 2] = self.action_p[action[:, 2]] * 0.1
        # p[:, 0] = action[:, 0]
        # p[:, 1] = action[:, 1]* 0.5
        # p[:, 2] = action[:, 2] * 0.1
        num1 = (np.sum(self.num_region) * p[:, 0]).astype(int)[0]
        num2 = (np.sum(self.num_region) * p[:, 1]).astype(int)[0]
        num3 = (np.sum(self.num_region) * p[:, 2]).astype(int)[0]
        # num1 = (self.num_region * 0).astype(int)
        # num2 = (self.num_region * 0.1).astype(int)
        # num3 = (self.num_region * 0).astype(int)

        self.tab_region_person['risk'] = self.prob_infecious + self.stat_prob
        self.tab_region_person['risk_infecious'] = self.prob_infecious

        index1=self.tab_region_person.nlargest(num1, 'risk')['pid'].to_numpy()
        index2=self.tab_region_person.nlargest(num2, 'risk_infecious')['pid'].to_numpy()
        index3=self.tab_region_person.nlargest(num3, 'risk_infecious')['pid'].to_numpy()
        

#        groups = self.tab_region_person.groupby('ranked_hzone')
#
#        index1, index2, index3 = zip(*[
#            (
#                group_data.nlargest(num1[group_name], 'risk')['pid'].to_numpy(),
#                group_data.nlargest(num2[group_name], 'risk_infecious')['pid'].to_numpy(),
#                group_data.nlargest(num3[group_name], 'risk_infecious')['pid'].to_numpy()
#            )
#            for group_name, group_data in groups
#        ])
#
#        index1 = np.concatenate(index1).astype(int)
#        index2 = np.concatenate(index2).astype(int)
#        index3 = np.concatenate(index3).astype(int)

#        groups = self.tab_region_person.groupby('ranked_hzone')

#        index1, index2, index3 = zip(*[
#            (
#                group_data.nlargest(num1[group_name], 'risk')['pid'].to_numpy(),
#                group_data.nlargest(num2[group_name], 'risk_infecious')['pid'].to_numpy(),
#                group_data.nlargest(num3[group_name], 'risk_infecious')['pid'].to_numpy()
#            )
#            for group_name, group_data in groups
#        ])

#        index1 = np.concatenate(index1).astype(int)
#        index2 = np.concatenate(index2).astype(int)
#        index3 = np.concatenate(index3).astype(int)
#
# # 2*action[action<=0]=1e-5
        individual_action = np.zeros(self.epc_env.n_agent)
        individual_action_real = np.zeros(self.epc_env.n_agent)
        bool_1 = np.zeros(self.epc_env.n_agent, dtype = bool)
        bool_1[index1] = True
        
        bool_2 = np.zeros(self.epc_env.n_agent, dtype = bool)
        bool_2[index2] = True
        
        bool_3 = np.zeros(self.epc_env.n_agent, dtype = bool)
        bool_3[index3] = True
        
        individual_action_real[bool_1] = 1
        individual_action_real[bool_2 & (self.prob_infecious > 0)] = 2
        individual_action_real[bool_3 & (self.prob_infecious > 0)] = 3
        
        individual_action[index1] = 1
        individual_action[index2] = 2
        individual_action[index3] = 3
        
        # p1 = action[:, 0][self.epc_env.tab_person_rank['ranked_hzone'].to_numpy()]
        # individual_action[self.prob > p1] = 1
        # p2 = action[:, 1][self.epc_env.tab_person_rank['ranked_hzone'].to_numpy()]
        # individual_action[self.prob > p2] = 2
        # p3 = action[:, 2][self.epc_env.tab_person_rank['ranked_hzone'].to_numpy()]
        # individual_action[self.prob > p3] = 3
        #        individual_action = np.zeros(self.epc_env.n_agent)
        #        # # # individual_action[self.prob<p1]=0
        #        # # # individual_action[(self.prob>=p1)&(self.prob<p2)]=1
        ##        individual_action[self.prob_infecious > 0.0001] = 1
        #        # # # # individual_action[(self.prob>=p2)&(self.prob<p3)]=2
        #        # # # # individual_action[self.prob>=p3]=3
        #        # # #        individual_action[self.prob > 0.0001] = 1
        ##        individual_action[self.prob_infecious > 0.001] = 1
        #        individual_action[self.prob_infecious > 0.00001] = 2
        #        individual_action[self.prob_infecious > 0.1] = 3
        # # individual_action[self.prob > 0.001] = 2
        # #
        #        individual_action_real=individual_action.copy()
        
        #
        # print("#########Action:",individual_2*action[:100])
        # individual_action「」
        # individual_2*action[self.prob>0.01]=2
        # individual_action = np.zeros(self.epc_env.n_agent)
        # individual_2*action[self.prob>0]=2
        # individual_2*action[self.prob>0]=1
        # individual_2*action[self.prob > 0] = 1
        # individual_2*action[self.prob>0]=3
        # individual_action=np.zeros(self.epc_env.n_agent)
        # individual_action[self.prob>0]=1
        
        self.sim_mat, obs, prob, region_infected, region_controlled = self.epc_env.step(self.sim_mat, individual_action_real)
        
        self.stat_prob = prob[0]
        self.prob_infecious = prob[1]
        
        done = False
        truncated = False  # 添加这行
        reward = self.get_reward(region_infected, region_controlled, individual_action)

        self.log_Q.append(np.sum(self.delta_Q_real))
        self.log_I.append(np.sum(self.delta_I))
        sim_date = self.epc_env.sim_date
        if sim_date == self.step_num:
            done = True
        info = {'Total_infect':np.sum(self.I), 'Total_quarantine':np.sum(self.Q),'Region_Q':np.mean(np.exp(
                            self.Q / self.num_region)),
                            'Region_I':np.mean(np.exp(
                            self.I / self.num_region)),
                            'daily_I':self.log_I,'daily_Q':self.log_Q,'Region_Q_log':self.Q}
        # if done==True:
        #     print('Total infected people:',np.sum(self.I))
        #     print('Total quarantine people:',np.sum(self.Q))
        return obs, reward, done, truncated, info


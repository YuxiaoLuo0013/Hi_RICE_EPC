
import time
import gym
from gym import spaces
from gym.envs.registration import register
import numpy as np
from torch.distributions.normal import Normal
from env.EPC_env.EpidemicSimulation_expert import EpidemicSimulation
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
                 max_iterday =150,
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
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (656, 2))
        # this_compartment, next_compartment, transit_countdown, immunity_type, immunity_days, infected_days, quarantine_type, quarantine_countdown, drug_access
        # self.action_space=spaces.Box(low=-np.inf,high=np.inf,shape=(587,3))
        self.action_space = spaces.MultiDiscrete([5 * 3] * 656)
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

        self.prob = prob
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

        reward = -(((20 * (self.delta_I)) ** 2.5 / self.num_region) + (self.delta_Q / self.num_region))

        return reward
    def step(self, action):
        individual_action = np.zeros(self.epc_env.n_agent)


        individual_action_real=individual_action.copy()

        self.sim_mat, self.sim_res, obs, prob, region_infected, region_controlled = self.epc_env.step(self.sim_mat,self.sim_res,individual_action_real)                                                                                       


        self.prob=prob

        done = False
        truncated = False  # 添加这行
        reward = self.get_reward(region_infected, region_controlled, individual_action)
        self.log_Q.append(np.sum(self.delta_Q_real))
        self.log_I.append(np.sum(self.delta_I))
        


        
        sim_date = self.epc_env.sim_date
        if sim_date == self.step_num:
            done = True
        info = {'Total_infect':np.sum(self.I), 'Total_quarantine':np.sum(self.Q),'Region_score':np.mean(np.exp(
                            self.Q / self.num_region)),
                            'daily_I':self.log_I,'daily_Q':self.log_Q}
        # if done==True:
        #     print('Total infected people:',np.sum(self.I))
        #     print('Total quarantine people:',np.sum(self.Q))
        return obs, reward, done, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

    def forward(self, x, action = None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, 0)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x)


def make_env(gym_id, seed):
    def thunk():
        env = gym.make(gym_id, seed = seed, autoreset = True)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs, -10, 10))
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


class Agent(nn.Module):
    def __init__(self, envs = None, hidden_size = 128):
        super(Agent, self).__init__()

        self.critic = nn.Sequential(
            layer_init(nn.Linear(16, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),

            layer_init(nn.Linear(hidden_size, 1), std = 1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(8, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 15), std = 0.01),
        )

    def forward(self, x_actor, x_critic, action = None):
        if len(x_actor.shape) == 3:
            logits = self.actor(x_actor).reshape(x_actor.shape[0], x_actor.shape[1], 3, 5)
            if action is None:
                action = torch.argmax(logits, dim = -1)
            logprob = Categorical(logits = logits).log_prob(action).reshape(x_actor.shape[0], x_actor.shape[1], -1)
            entropy = Categorical(logits = logits).entropy().reshape(x_actor.shape[0], x_actor.shape[1], -1)
            return action, logprob.sum(-1), entropy.sum(-1), self.critic(x_critic)
        elif len(x_actor.shape) == 2:
            logits = self.actor(x_actor).reshape(x_actor.shape[0], 3, 5)
            if action is None:
                action = torch.argmax(logits, dim = -1)
            #                action = Categorical(logits = logits).sample()
            logprob = Categorical(logits = logits).log_prob(action).reshape(x_actor.shape[0], -1)
            entropy = Categorical(logits = logits).entropy().reshape(x_actor.shape[0], -1)
            return action, logprob.sum(-1), entropy.sum(-1), self.critic(x_critic)


if __name__ == "__main__":

    register(
        id = 'EpidemicModel-v0',  # 使用一个唯一的ID
        entry_point = 'env_ma:EpidemicModel',  # 替换为您的环境类路径
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

        prob_log = np.array([])
        prob_mean_log = []
        for i in range(150):
            print("Days:", i)
            # print(obs[obs != 0])
            # obs=(obs-mean)/(variance + 1e-8)
            # action, logprob, _, value=model(torch.tensor(obs[:,:8]).float().cuda(),torch.tensor(obs).float().cuda())
            # action=np.c_[np.ones((587,1)),np.ones((587,3))]
            # zone_2*action[obs > 1] = 2
            # zone_2*action[obs>100]=2
            # print("zone action:",zone_action.mean())
            # action=zone_2*action[env.epc_env.tab_person_rank['ranked_hzone'].to_numpy()]
            # action=np.concatenate((-np.ones((587,2)),np.ones((587,2))),axis=1)
            # action=action.cpu().detach().numpy()
            # action = np.random.choice([0,1,2], size=env.epc_env.n_agent)
            # action = np.zeros(env.epc_env.n_agent)
            # action=np.zeros(env.epc_env.n_agent)
            action = 1 * np.ones((587, 3)).astype(int)
            next_obs, reward, done, truncated, info = envs.step(action)
            obs = next_obs
            #     print(done)
            #     print(np.mean(reward))
            print("delta I", np.sum(envs.delta_I))
            print("delta Q", np.sum(envs.delta_Q_real))

        # #     print(env.sim_res['daily_infect'])
        print('Total infected people:', sum(envs.sim_res['daily_infect']))
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


# -*- coding = utf-8 -*-
# @time:2024/6/14 20:53
# Author:Yuxiao
# @File:train_HRLI.py
# -*- coding = utf-8 -*-
# @time:2024/3/8 16:57
# Author:Yuxiao
# @File:1.py
import argparse
import os
import random
import time
from distutils.util import strtobool
from torch.distributions.categorical import Categorical
from torch.distributions import Normal
import torch.nn.functional as F
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
from normalization import Normalization
from agent import Agent, Agent_lstm,Agent_gcnlstm
from collections import deque
warnings.filterwarnings("ignore")
register(
    id = 'EpidemicModel-v0',  # 使用一个唯一的ID
    entry_point = 'env.env_HRLI:EpidemicModel',  # 替换为您的环境类路径
    # 这里可以添加更多的参数，如max_episode_steps等
)


# register(
#         id = 'EpidemicModel-v0',  # 使用一个唯一的ID
#         entry_point = 'env.test_env:test_env',  # 替换为您的环境类路径
#         # 这里可以添加更多的参数，如max_episode_steps等
#     )

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type = str, default = os.path.basename(__file__).rstrip(".py"),
                        help = "the name of this experiment")
    parser.add_argument("--gym-id", type = str, default = "EpidemicModel-v0",
                        help = "the id of the gym environment")
    parser.add_argument("--learning-rate", type = float, default = 1e-4,
                        help = "the learning rate of the optimizer")
    parser.add_argument("--seed", type = int, default = 1,
                        help = "seed of the experiment")
    parser.add_argument("--total-timesteps", type = int, default = 1000,
                        help = "total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type = lambda x:bool(strtobool(x)), default = True, nargs = "?",
                        const = True,
                        help = "if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type = lambda x:bool(strtobool(x)), default = False, nargs = "?", const = True,
                        help = "if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type = str, default = "ppo-implementation-details",
                        help = "the wandb's project name")
    parser.add_argument("--wandb-entity", type = str, default = None,
                        help = "the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type = lambda x:bool(strtobool(x)), default = False, nargs = "?",
                        const = True,
                        help = "weather to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type = int, default = 1,
                        help = "the number of parallel game environments")
    parser.add_argument("--num-steps", type = int, default = 20,
                        help = "the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "Use GAE for advantage computation")
    parser.add_argument("--gamma", type = float, default = 0.99,
                        help = "the discount factor gamma")
    parser.add_argument("--gae-lambda", type = float, default = 0.95,
                        help = "the lambda for the general advantage estimation")
    parser.add_argument("--num_minibatches", type = int, default = 16,
                        help = "the number of mini-batches")
    parser.add_argument("--update-epochs", type = int, default = 4,
                        help = "the K epochs to update the policy")
    parser.add_argument("--norm-adv", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "Toggles advantages normalization")
    parser.add_argument("--clip-coef", type = float, default = 0.1,
                        help = "the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type = float, default = 0.01,
                        help = "coefficient of the entropy")
    parser.add_argument("--vf-coef", type = float, default = 0.5,
                        help = "coefficient of the value function")
    parser.add_argument("--max-grad-norm", type = float, default = 0.5,
                        help = "the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type = float, default = 0.2,
                        help = "the target KL divergence threshold")
    parser.add_argument("--use-gpu", type = lambda x:bool(strtobool(x)), default = True, nargs = "?", const = True,
                        help = "Toggles GPU utilization")
    parser.add_argument("--gpus", type = int, nargs = "+", default = [0],
                        help = "the gpus to use")

    # test
    parser.add_argument("--num-test", type = int, default = 5,
                        help = "the number of parallel test game environments")
    parser.add_argument("--test-seed", type = int, default = 123456,
                        help = "seed of the test experiment")
    parser.add_argument("--num-pop", type = int, default = 1618304,
                        help = "the number of population")
    parser.add_argument("--num-days", type = int, default = 150,
                        help = "the number of days for testing")
    parser.add_argument("--p-trans", type = float, default = 0.15,
                        help = "the probability of transmission")

    args = parser.parse_args()
    # args.num_sample=128
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = 4
    # fmt: on
    return args


def make_env(gym_id, test, seed,p_trans):
    env = gym.make(gym_id, if_test = test, seed = seed, autoreset = False, ptrans = p_trans)
    # env = gym.wrappers.NormalizeObservation(env)
    # env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs, -10, 10))
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


# def Agent as 2 layer mlp
class Agent_PPO(nn.Module):
    def __init__(self, hidden_size = 64,input_size=10):
        super(Agent_PPO, self).__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 4),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, 4)*0.1)

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    def forward(self, x_actor,x_critic,action=None):
        action_mean = self.actor_mean(x_actor)
        action_std = self.actor_logstd.expand_as(action_mean)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x_critic)
    
class Agent_DQN(nn.Module):
    def __init__(self, hidden_size=64, input_size=2):
        super(Agent_DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3)  # 3 outputs for -1, 0, 1
        )
    
    def forward(self, x):
        return self.network(x)
    
    def act(self, state, epsilon=0.0):
        if random.random() > epsilon:
            with torch.no_grad():
                q_values = self.forward(state)
                return q_values.max(1)[1].view(1, 1)  # Returns index of max Q-value
        else:
            return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)


def compute_td_loss(agent, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    q_values = agent(states)
    next_q_values = agent(next_states)
    
    q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = rewards + gamma * next_q_value * (1 - dones)
    
    loss = nn.MSELoss()(q_value, expected_q_value.detach())
    return loss



if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project = args.wandb_project_name,
            entity = args.wandb_entity,
            sync_tensorboard = True,
            config = vars(args),
            name = run_name,
            monitor_gym = True,
            save_code = True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # env setup
    envs = make_env(args.gym_id, test = False, seed = args.seed, p_trans = args.p_trans)
    n_pop = args.num_pop
    print("Sucessfully initialized", args.num_envs, "environments")
    agent_ind = Agent_PPO( hidden_size = 64,input_size=10).to(device)
    agent_res=Agent_DQN(hidden_size=16,input_size=2).to(device)

    optimizer = optim.Adam(agent_ind.parameters(), lr = args.learning_rate, eps = 1e-5)
    optimizer_res = optim.Adam(agent_res.parameters(), lr = args.learning_rate, eps = 1e-5)
    # ALGO Logic: Storage setup
    # args.num_envs=587
    obs = torch.zeros((args.num_steps,envs.observation_space[1].shape[0],envs.observation_space[1].shape[1])).to(device)
    actions = torch.zeros((args.num_steps, envs.action_space.shape[0], 4)).to(device)
    logprobs = torch.zeros((args.num_steps, envs.observation_space[1].shape[0])).to(device)
    rewards = torch.zeros((args.num_steps, envs.observation_space[1].shape[0])).to(device)
    dones = torch.zeros((args.num_steps,1)).to(device)
    values = torch.zeros((args.num_steps,  envs.observation_space[1].shape[0])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs= envs.reset()[0]
    next_obs=next_obs[0]
    res_obs=next_obs[1]
    hzone_id=envs.epc_env.tab_region_person['ranked_hzone']
    next_obs = torch.Tensor(next_obs).to(device)
    res_obs = torch.Tensor(res_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    reward_log = []
    log_I = []
    log_Q = []
    num_train_time = 0
    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):

            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            # Get DQN action from res_obs
            with torch.no_grad():
                res_action = agent_res.get_action(res_obs)
            res_action=res_action.detach().cpu().numpy()
            res_action1=np.zeros((args.num_pop,args.num_envs))
            res_action1=res_action[self.hzone_id,:]
            res_action1=res_action1.reshape(args.num_envs,args.num_pop,1)
            # Convert res_action to appropriate format if necessary
            # For example, if res_action is an index, you might need to convert it to one-hot encoding
            # res_action_onehot = F.one_hot(res_action, num_classes=4)
            # ALGO LOGIC: action logic
            next_obs[:,:,-1]=res_action1
            with torch.no_grad():
                action, logprob, _, value = agent_ind(next_obs, next_obs)
                values[step] = value.flatten(start_dim = 0)
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())

            res_obs=next_obs[1]
            res_obs = torch.Tensor(res_obs).to(device)
            res_reward=torch.Tensor(reward[1]).to(device)
            # Update DQN using res_reward and res_obs
            # Get current Q values
            current_q_values = agent_ind.q_network(res_obs)
            
            # Get next state Q values
            with torch.no_grad():
                next_q_values = agent_ind.target_network(res_obs).max(1)[0]
            
            # Compute target Q values
            target_q_values = res_reward + args.gamma * next_q_values * (1 - next_done)
            
            # Compute loss
            loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
            
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update target network if it's time
            if global_step % args.target_network_frequency == 0:
                agent_ind.update_target()

            next_obs = torch.Tensor(next_obs[0]).to(device)
            # reward= reward_scaling(reward)
            # reward=reward.reshape(-1,1).repeat(envs.single_observation_space.shape[1],axis=1)
            rewards[step] = torch.tensor(reward[0].repeat(envs.observation_space[1].shape[0])).to(device)
            next_done=torch.Tensor([done]).to(device)
            reward_log.append(reward[0])

            if done == True:
                num_train_time += 1
                print('Total_infect:', info['Total_infect'])
                print('Total_quarantine:', info['Total_quarantine'])
                print('Global Step:', global_step)
                print('Episode reward:', np.sum(reward_log))
                reward_log = []
                next_obs = envs.reset()[0][0]
                next_obs = torch.Tensor(next_obs).to(device)

        with torch.no_grad():
            _, _, _, next_value = agent_ind(next_obs, next_obs)
            next_value = next_value.flatten(start_dim = 0)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal.unsqueeze(1) - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal.unsqueeze(
                        1) * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        # random_indices = np.random.choice(np.arange(envs.single_observation_space.shape[1]), size=args.num_sample)
        # b_obs=obs.reshape((-1,envs.single_observation_space.shape[1],envs.single_observation_space.shape[1]))
        # b_logprobs = logprobs.reshape(-1,envs.single_observation_space.shape[1])
        # b_actions = actions.reshape((-1,envs.single_observation_space.shape[1],envs.single_action_space.shape[1]))
        # b_advantages = advantages.reshape(-1,envs.single_observation_space.shape[1])
        # b_returns = returns.reshape(-1,envs.single_observation_space.shape[1])
        # b_values = values.reshape(-1,envs.single_observation_space.shape[1])
#        b_obs = obs.permute(0, 1, 3, 2, 4).reshape(
#            (-1, envs.single_observation_space.shape[0], envs.single_observation_space.shape[2]))
#        b_logprobs = logprobs.reshape(-1)
#        b_actions = actions.reshape((-1, 3))
#        b_advantages = advantages.reshape(-1)
#        b_returns = returns.reshape(-1)
#        b_values = values.reshape(-1)

        b_obs = obs.reshape((-1, envs.observation_space[1].shape[0], envs.observation_space[1].shape[1]))
        b_logprobs = logprobs.reshape(-1, envs.observation_space[1].shape[0])
        b_actions = actions.reshape((-1, envs.observation_space[1].shape[0],4))
        b_advantages = advantages.reshape(-1, envs.observation_space[1].shape[0])
        b_returns = returns.reshape(-1, envs.observation_space[1].shape[0])
        b_values = values.reshape(-1, envs.observation_space[1].shape[0])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent_ind(b_obs[mb_inds], b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.squeeze(2)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent_ind.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        # save model when global step 1000 and creat checkpoint file if not exist
        # if global_step % 2400 == 0:
        #     if not os.path.exists("./checkpoint"):
        #         os.mkdir("./checkpoint")
        #     if not os.path.exists("./running_parameters"):
        #         os.mkdir("./running_parameters")
        #     torch.save(agent.state_dict(), f"./checkpoint/{run_name}_{global_step}.pth")
        #     np.save(f"./running_parameters/{run_name}_mean.npy", Normalization.running_ms.mean)
        #     np.save(f"./running_parameters/{run_name}_std.npy", Normalization.running_ms.std)

        # save model when global step 1000 and creat checkpoint file if not exist
    #        if global_step % (args.num_days*3*8)  == 0:
    #            print('Start Testing!')
    #            agent.test=True
    #            with torch.no_grad():
    #                next_obs_test = envs_test.reset()[0]
    #                next_obs_test = (next_obs_test -Normalization.running_ms.mean) / (Normalization.running_ms.std + 1e-8)
    #                for update in range(0, args.num_days):
    #                    next_obs_test = torch.Tensor(next_obs_test).to(device)
    #                    action, logprob, _, value = agent(next_obs_test[:, :, :, :8], next_obs_test)
    #                    next_obs_test, reward, done, _, info = envs_test.step(action.cpu().numpy())
    #                    next_obs_test = (next_obs_test - Normalization.running_ms.mean) / (Normalization.running_ms.std + 1e-8)
    #                info_log = {}
    #                info_log['Total_infect'] = []
    #                info_log['Total_quarantine'] = []
    #                for i in range(args.num_test):
    #                    info_log['Total_infect'].append(info['final_info'][i]['Total_infect'])
    #                    info_log['Total_quarantine'].append(info['final_info'][i]['Total_quarantine'])
    #                test_score1=np.mean(np.exp(np.array(info_log['Total_infect'])*20/n_pop)+np.exp(np.array(info_log['Total_quarantine'])/n_pop))
    #                print('Mean total infect :', np.mean(info_log['Total_infect']))
    #                print('Mean total quarantine :', np.mean(info_log['Total_quarantine']))
    #                print('Testing score:',test_score1)
    #                if test_score1<test_score:
    #                    print('Update the best model parameters')
    #                    if not os.path.exists("./checkpoint"):
    #                        os.mkdir("./checkpoint")
    #                    if not os.path.exists("./running_parameters"):
    #                        os.mkdir("./running_parameters")
    #                    torch.save(agent.state_dict(), f"./checkpoint/{run_name}_best_model.pth")
    #                    np.save(f"./running_parameters/{run_name}_mean.npy", Normalization.running_ms.mean)
    #                    np.save(f"./running_parameters/{run_name}_std.npy", Normalization.running_ms.std)
    #                    test_score=test_score1
    #            agent.test = False
    # writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    # envs_test.close()
    writer.close()

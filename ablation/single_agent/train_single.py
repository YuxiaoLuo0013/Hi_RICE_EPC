# -*- coding = utf-8 -*-
# @time:2024/3/8 16:57
# Author:Yuxiao
# @File:1.py
import argparse
import os
import random
import time
from distutils.util import strtobool

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

warnings.filterwarnings("ignore")
register(
    id = 'EpidemicModel-v0',  # 使用一个唯一的ID
    entry_point = 'env.env_single:EpidemicModel',  # 替换为您的环境类路径
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
    parser.add_argument("--total-timesteps", type = int, default = 40000,
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
    parser.add_argument("--wtp", type = float, default = 1.0,
                        help = "alpha in reward")
    parser.add_argument("--wta", type = float, default = 1.0,
                        help = "alpha in reward")
    parser.add_argument("--alpha", type = float, default = 2.0,
                        help = "alpha in reward")
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
    parser.add_argument("--gpu", type = int, default = 0,
                        help = "the gpu to use")

    # test

    parser.add_argument("--num-test", type = int, default = 5,
                        help = "the number of parallel test game environments")
    parser.add_argument("--test-seed", type = int, default = 123456,
                        help = "seed of the test experiment")
    parser.add_argument("--num-pop", type = int, default = 1618304,
                        help = "the number of population")
    parser.add_argument("--num-days", type = int, default = 150,
                        help = "the number of days for testing")

    parser.add_argument("--p-trans", type = float, default = 0.2,
                        help = "p of trans")

    args = parser.parse_args()

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = 4 * args.num_envs
    # fmt: on
    return args


def make_env(gym_id, test, p_trans, alpha, seed):
    def thunk():
        env = gym.make(gym_id, if_test = test, seed = seed, autoreset = False, ptrans = p_trans, alpha = alpha)
        # env = gym.wrappers.NormalizeObservation(env)
        # env = gym.wrappers.TransformObservation(env, lambda obs:np.clip(obs, -10, 10))
        env = gym.wrappers.FrameStack(env, 7)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}_{args.p_trans}_single"
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
#    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.manual_seed(args.seed)
#    torch.backends.cudnn.deterministic = True
#    torch.backends.cudnn.benchmark = False
#    torch.backends.cudnn.enabled = False
#    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
#    torch.use_deterministic_algorithms(True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.use_gpu else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.gym_id, False,args.p_trans,args.alpha,args.seed + int(i * args.total_timesteps / args.num_envs) + args.num_envs) for i
         in range(args.num_envs)]
    )
    envs_test = gym.vector.AsyncVectorEnv(
       [make_env(args.gym_id,True, args.p_trans,args.alpha,args.test_seed + int(i + args.num_test)) for i in range(args.num_test)])
    test_score=1e6
    # reward_scaling = RewardScaling(shape = args.num_envs, gamma = args.gamma)
    Normalization = Normalization(shape = (1, 22))
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # print sucessfully initialized num_envs env
    # envs = gym.wrappers.NormalizeObservation(envs)
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)
    n_pop = args.num_pop
    print("Sucessfully initialized", args.num_envs, "environments")
    print("Sucessfully initialized", args.num_test, "test environments")
    agent = Agent_lstm(envs, hidden_size = 128).to(device)
    #    agent.load_state_dict(
    #        torch.load('/EPC_RF/checkpoint/EpidemicModel-v0__train_vanilla_PPO__1__1709717936_best_model.pth'))
#    if args.use_gpu and len(args.gpus) > 1:
#        print("Using", torch.cuda.device_count(), "GPUs!")
#        agent = nn.DataParallel(agent, device_ids = args.gpus)
    #    Normalization.running_ms.mean=np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1709717936_mean.npy')
    #    Normalization.running_ms.std=np.load('/EPC_RF/running_parameters/EpidemicModel-v0__train_vanilla_PPO__1__1709717936_std.npy')
    optimizer = optim.Adam(agent.parameters(), lr = args.learning_rate, eps = 1e-5)
    # ALGO Logic: Storage setup
    # args.num_envs=587
    obs = torch.zeros((args.num_steps, args.num_envs) + (envs.single_observation_space.shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, envs.single_action_space.shape[0], 3)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, envs.single_observation_space.shape[1])).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, envs.single_observation_space.shape[1])).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, envs.single_observation_space.shape[1])).to(device)
    indices = ((torch.arange(envs.single_observation_space.shape[1]).unsqueeze(0) + torch.arange(
        envs.single_observation_space.shape[1]).unsqueeze(1)) % envs.single_observation_space.shape[1]).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = envs.reset()[0]
    next_obs = Normalization(next_obs)
    next_obs = torch.Tensor(next_obs).to(device)
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
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent(next_obs[:, :, :, 11:], next_obs[:, :, :, 11:])
                values[step] = value.flatten(start_dim = 1)
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
            next_obs = Normalization(next_obs)
            next_obs = torch.Tensor(next_obs).to(device)
            # reward= reward_scaling(reward)
            # reward=reward.reshape(-1,1).repeat(envs.single_observation_space.shape[1],axis=1)
            rewards[step] = torch.tensor(reward).to(device)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            reward_log.append(np.mean(reward))

            if done[0] == True:
                num_train_time += 1
                for i in range(args.num_envs):
                    print('Total_infect:', info['final_info'][i]['Total_infect'])
                    print('Total_quarantine:', info['final_info'][i]['Total_quarantine'])
                print('Global Step:', global_step)
                print('Episode reward:', np.sum(reward_log))
                reward_log = []
                next_obs = envs.reset()[0]
                next_obs = Normalization(next_obs)
                next_obs = torch.Tensor(next_obs).to(device)

                if (num_train_time % 3 == 0)&(num_train_time>30):
                    print('Start Testing!')
                    agent.test = True
                    with torch.no_grad():
                        next_obs_test = envs_test.reset()[0]
                        next_obs_test = (next_obs_test - Normalization.running_ms.mean) / (
                                    Normalization.running_ms.std + 1e-8)
                        for update in range(0, args.num_days):
                            next_obs_test = torch.Tensor(next_obs_test).to(device)
                            action, logprob, _, value = agent(next_obs_test[:, :, :, 11:], next_obs_test[:, :, :, 11:])
                            next_obs_test, reward, done, _, info = envs_test.step(action.cpu().numpy())
                            next_obs_test = (next_obs_test - Normalization.running_ms.mean) / (
                                        Normalization.running_ms.std + 1e-8)
                        info_log = {}
                        info_log['Total_infect'] = []
                        info_log['Total_quarantine'] = []
                        for i in range(args.num_test):
                            info_log['Total_infect'].append(info['final_info'][i]['Total_infect'])
                            info_log['Total_quarantine'].append(info['final_info'][i]['Total_quarantine'])
                        final_test_score = np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / n_pop) + np.exp(
                            np.array(info_log['Total_quarantine']) / n_pop))
                        print('Mean total infect :', np.mean(info_log['Total_infect']))
                        print('Mean total quarantine :', np.mean(info_log['Total_quarantine']))
                        print('Testing score:', final_test_score)
                        if np.mean(info_log['Total_infect']) < test_score:
                            print('Update the best model parameters')
                            if not os.path.exists("./checkpoint"):
                                os.mkdir("./checkpoint")
                            if not os.path.exists("./running_parameters"):
                                os.mkdir("./running_parameters")
                            torch.save(agent.state_dict(), f"./checkpoint/{run_name}_best_model.pth")
                            np.save(f"./running_parameters/{run_name}_mean.npy", Normalization.running_ms.mean)
                            np.save(f"./running_parameters/{run_name}_std.npy", Normalization.running_ms.std)
                            test_score = np.mean(info_log['Total_infect'])
                    agent.test = False

        # bootstrap value if not done
        with torch.no_grad():
            _, _, _, next_value = agent(next_obs[:, :, :, 11:], next_obs[:, :, :, 11:])
            next_value = next_value.flatten(start_dim = 1)
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

        b_obs = obs.reshape((-1, envs.single_observation_space.shape[0], envs.single_observation_space.shape[1], envs.single_observation_space.shape[2]))
        b_logprobs = logprobs.reshape(-1, envs.single_observation_space.shape[1])
        b_actions = actions.reshape((-1, envs.single_observation_space.shape[1],3))
        b_advantages = advantages.reshape(-1, envs.single_observation_space.shape[1])
        b_returns = returns.reshape(-1, envs.single_observation_space.shape[1])
        b_values = values.reshape(-1, envs.single_observation_space.shape[1])

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent(b_obs[mb_inds, :,:, 11:], b_obs[mb_inds, :,:, 11:], b_actions[mb_inds])
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
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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

    envs.close()
    envs_test.close()
    writer.close()
    #test
    seed = 2024
    num_envs = 10
    use_gpu = True
    random.seed(seed)
    np.random.seed(seed)
    mean = np.load(f"./running_parameters/{run_name}_mean.npy")
    variance = np.load(f"./running_parameters/{run_name}_std.npy")
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env('EpidemicModel-v0', True,args.p_trans,args.alpha, seed + int(i + num_envs)) for i in range(num_envs)]
    )
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    # print sucessfully initialized num_envs env
    # envs = gym.wrappers.NormalizeObservation(envs)
    # envs = gym.wrappers.RecordEpisodeStatistics(envs)

    print("Sucessfully initialized", num_envs, "environments")

    agent = Agent_lstm(envs, hidden_size = 128).to(device)
    agent.test = True
    agent.load_state_dict(
        torch.load(f"./checkpoint/{run_name}_best_model.pth"))
    agent.eval()
    next_obs = envs.reset()[0]
    next_obs = (next_obs - mean) / (variance + 1e-8)
    for update in range(0, 150):
        next_obs = torch.Tensor(next_obs).to(device)
        action, logprob, _, value = agent(next_obs[:, :, :, 11:], next_obs[:, :, :, 11:])
        next_obs, reward, done, _, info = envs.step(action.cpu().numpy())
        next_obs = (next_obs - mean) / (variance + 1e-8)
    
    info_log = {}
    info_log['Total_infect'] = []
    info_log['Total_quarantine'] = []
    info_log['Region_cI'] = []
    info_log['Region_cQ'] = []
    Region_Q_log = np.zeros(656)
    daily_I = np.zeros(args.num_days)
    daily_Q = np.zeros(args.num_days)
    for i in range(num_envs):
        info_log['Total_infect'].append(info['final_info'][i]['Total_infect'])
        info_log['Total_quarantine'].append(info['final_info'][i]['Total_quarantine'])
        info_log['Region_cI'].append(info['final_info'][i]['Region_I'])
        info_log['Region_cQ'].append(info['final_info'][i]['Region_Q'])
        daily_I = np.vstack((daily_I, info['final_info'][i]['daily_I']))
        daily_Q = np.vstack((daily_Q, info['final_info'][i]['daily_Q']))
        Region_Q_log = np.vstack((Region_Q_log, info['final_info'][i]['Region_Q_log']))
        print('Total infect :', info['final_info'][i]['Total_infect'])
        print('Total quarantine :', info['final_info'][i]['Total_quarantine'])
    print('Mean total infect :', np.mean(info_log['Total_infect']))

    if not os.path.exists(daily_record_dir):
        os.makedirs(daily_record_dir)
        print(f"Created directory: {daily_record_dir}")
    np.save(f"./daily_record/{args.p_trans}_alpha_{args.alpha}_wta_{args.wta}_wtp_{args.wtp}_daily_I_single.npy", daily_I[1:, :])
    np.save(f"./daily_record/{args.p_trans}_alpha_{args.alpha}_wta_{args.wta}_wtp_{args.wtp}_daily_Q_single.npy", daily_Q[1:, :])
    np.save(f"./daily_record/{args.p_trans}_alpha_{args.alpha}_wta_{args.wta}_wtp_{args.wtp}_Region_Q_single.npy", Region_Q_log[1:,:])


    print('Mean total quarantine :', np.mean(info_log['Total_quarantine']))
    print('Mean region I:', np.mean(info_log['Region_cI']))
    print('Mean region Q:', np.mean(info_log['Region_cQ']))
    print('Final score:', np.mean(np.exp(np.array(info_log['Total_infect']) * 20 / n_pop) + np.exp(
        np.array(info_log['Total_quarantine']) / n_pop)))
                           
    

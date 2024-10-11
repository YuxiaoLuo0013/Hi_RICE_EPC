# -*- coding = utf-8 -*-
# @time:2024/6/17 20:37
# Author:Yuxiao
# @File:train_GBM.py
import numpy as np
import time
import gym
from gym import register
from catboost import CatBoostClassifier
from collections import deque
import tqdm
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import accuracy_score, recall_score, precision_score

class asym_class_buffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen = capacity)
    def push(self,state, label):
        state = np.expand_dims(state, 0)
        self.buffer.append((state, label))

    def store_sample_trans(self, state, label):
        random_index = random.sample(range(len(state)), 5000)
        state_sample = state[random_index]
        label_sample = label[random_index]
        del state, label
        for i in range(len(label_sample)):
            state1 = state_sample[i]
            label1 = label_sample[i]
            self.push(state1, label1)
    def __len__(self):
        return len(self.buffer)
    def save_to_npz(self, file_name):
        # Convert the deque to a list of arrays
        data_list = list(self.buffer)

        # Assuming all elements in deque have the same structure and size
        # For more complex scenarios, you might need a different approach

        states = np.array([item[0] for item in data_list])
        labels = np.array([item[1] for item in data_list])
        # Saving states and labels to a single .npz file
        np.savez(file_name,states = states, labels = labels)



# 定义和训练CatBoost模型
def train_catboost(X_train, y_train, depth=6, learning_rate=0.1, iterations=1000, class_weights=[1, 1]):
    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        verbose=0,
        class_weights=class_weights
    )
    model.fit(X_train, y_train)
    return model


# 预测函数
def predict(model, X):
    return model.predict(X)


# 模型选择函数
def predict_with_threshold(model, X, threshold=0.5):
    probabilities = model.predict_proba(X)[:, 1]
    return (probabilities >= threshold).astype(int)

# 模型选择函数
def select_model_and_predict(high_precision_model, high_recall_model, X_val, I_new, gamma, recall_threshold=0.5):
    if I_new <= gamma:
        # print('presion')
        return predict_with_threshold(high_precision_model, X_val, threshold=recall_threshold)
    else:
        # print('recall')
        return predict_with_threshold(high_recall_model, X_val, threshold=recall_threshold)



if __name__ == "__main__":
    p_trans=[0.07,0.15,0.24]
    register(
        id = 'EpidemicModel-v0',  # 使用一个唯一的ID
        entry_point = 'env.env_GBM:EpidemicModel',  # 替换为您的环境类路径
        # 这里可以添加更多的参数，如max_episode_steps等
    )
    for p_tran in p_trans:
        envs = gym.make('EpidemicModel-v0',ptrans =p_tran)
        buffer=asym_class_buffer(150000)
        # high_precision_model = init_model(class_weights = [1, 80])
        #
        # # 训练高召回率模型 (w0=1, w1=15)
        # high_recall_model = init_model(class_weights = [1, 15])

        # 开始模拟
        for i in range(1):
            obs, info = envs.reset()
            start_time = time.time()
            for ii in range(150):
                # print("Days:", ii)

                if ii % 5 == 0:
                    label = np.zeros((envs.epc_env.n_agent)).astype(int)
                    state=obs
                action = 0 * np.ones((envs.epc_env.n_agent)).astype(int)
                next_obs, reward, done, truncated, info = envs.step(action)
                label[(envs.sim_mat['comp_this']>1) & (envs.sim_mat['comp_this']<10)] = 1
                if (ii % 5 == 0) & (ii != 0):
                    buffer.store_sample_trans(state, label)
                obs = next_obs
            buffer.save_to_npz(f'./GBM_buffer_{p_tran}.npz')
#
        data=np.load(f'./GBM_buffer_{p_tran}.npz')
        #split test and train
        X_train, X_test, y_train, y_test = train_test_split(data['states'].reshape(-1,11), data['labels'], test_size=0.2, random_state=42)
        #train model
        high_precision_model = train_catboost(X_train, y_train, class_weights=[1,15])
        high_recall_model = train_catboost(X_train, y_train, class_weights=[1, 150])
        #print accuracy
        y_pred = predict(high_precision_model, X_test)
        print('Precision:', precision_score(y_test, y_pred))
        print('Recall:', recall_score(y_test, y_pred))

        y_pred = predict(high_recall_model, X_test)
        print('Precision:', precision_score(y_test, y_pred))
        print('Recall:', recall_score(y_test, y_pred))


        #test
        envs = gym.make('EpidemicModel-v0',ptrans =p_tran)
        # high_precision_model = init_model(class_weights = [1, 80])
        #
        # # 训练高召回率模型 (w0=1, w1=15)
        # high_recall_model = init_model(class_weights = [1, 15])
        recall_threshold = 0.2
        gamma = 0.0001
        # start test
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

                # 获取当前环境的特征作为输入
                X_val = obs

                # 假设当前新感染率
                I_new = np.sum(envs.delta_I) / envs.epc_env.n_agent

                # 使用预测模型选择最佳行动
                predictions = select_model_and_predict(high_precision_model, high_recall_model, X_val, I_new, gamma,recall_threshold)
                action = predictions.reshape(envs.epc_env.n_agent)
                # print(np.where(action==1)[0])
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


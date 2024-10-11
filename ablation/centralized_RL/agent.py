# -*- coding = utf-8 -*-
# @time:2023/12/13 14:44
# Author:Yuxiao
# @File:agent.py
import math, random
import gym
from torch.distributions import Normal
from torch_geometric.data import Data, Batch
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GATConv
import numpy as np
import torch
from torch import nn
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class EinsteinLayer(nn.Module):
    def __init__(self, N, in_features, out_features):
        super(EinsteinLayer, self).__init__()
        self.N = N
        self.in_features = in_features
        self.out_features = out_features
        self.weights = nn.Parameter(torch.randn(N, in_features, out_features))
        self.biases = nn.Parameter(torch.randn(N, out_features))

    def forward(self, x):
        return torch.einsum('bnd,nmd->bnm', x, self.weights) + self.biases
class Agent(nn.Module):
    def __init__(self, envs, hidden_size = 256):
        super(Agent, self).__init__()

        self.critic_mlp = nn.Sequential(
            nn.Flatten(start_dim = 1),
            layer_init(nn.Linear(envs.single_observation_space.shape[2] * 7, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.critic = layer_init(nn.Linear(hidden_size, 1), std = 1.0)
        self.actor_mlp = nn.Sequential(
            nn.Flatten(start_dim = 1),
            layer_init(nn.Linear(int((envs.single_observation_space.shape[2] - 8) * 7), hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        self.actor = layer_init(nn.Linear(hidden_size, envs.single_action_space[0].n), std = 0.01)

    def forward(self, x_actor, x_critic, action = None):
        squeeze = False
        if len(x_actor.shape) == 3:
            squeeze = True
            x_actor = x_actor.unsqueeze(2)
            x_critic = x_critic.unsqueeze(2)
            action = action.unsqueeze(1)
        B, L, M, H = x_actor.shape
        B1, L1, M1, H1 = x_critic.shape
        x_actor = x_actor.permute(1, 0, 2, 3).reshape(L, B * M, H).permute(1, 0, 2)
        x_critic = x_critic.permute(1, 0, 2, 3).reshape(L1, B1 * M1, H1).permute(1, 0, 2)
        actor_hidden = self.actor_mlp(x_actor)
        actor_hidden = actor_hidden
        critic_hidden = self.critic_mlp(x_critic)
        critic_hidden = critic_hidden.reshape(B1, M1, -1)
        logits = self.actor(actor_hidden).reshape(B, M, 3, 4)
        if action is None:
            action = Categorical(logits = logits).sample()
        logprob = Categorical(logits = logits).log_prob(action)
        entropy = Categorical(logits = logits).entropy()
        if squeeze == True:
            logprob = logprob.squeeze(1)
            entropy = entropy.squeeze(1)
            critic_hidden = critic_hidden.squeeze(1)
        return action, logprob.sum(-1), entropy.sum(-1), self.critic(critic_hidden)
class Agent_lstm(nn.Module):
    def __init__(self, envs,hidden_size=64):
        super(Agent_lstm, self).__init__()
        self.test=False
        self.critic_GRU = nn.Sequential(
            layer_init(nn.Linear(envs.single_observation_space.shape[2]-11, hidden_size)),
            nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True),
        )
        self.critic=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std = 1.0))
        self.actor_GRU = nn.Sequential(
            layer_init(nn.Linear(int(envs.single_observation_space.shape[2]-11), hidden_size)),
            nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True),
        )
        self.actor=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.single_action_space[0].n), std = 0.01))
    def forward(self, x_actor,x_critic,action=None):
        squeeze=False
        if len(x_actor.shape)==3:
            squeeze=True
            x_actor=x_actor.unsqueeze(2)
            x_critic=x_critic.unsqueeze(2)
            action=action.unsqueeze(1)
        B,L,M,H=x_actor.shape
        B1, L1, M1, H1 = x_critic.shape
        x_actor=x_actor.permute(1,0,2,3).reshape(L,B*M,H).permute(1,0,2)
        x_critic=x_critic.permute(1,0,2,3).reshape(L1,B1*M1,H1).permute(1,0,2)
        actor_hidden,_=self.actor_GRU(x_actor)
        actor_hidden=actor_hidden[:,-1,:]
        critic_hidden,_=self.critic_GRU(x_critic)
        critic_hidden=critic_hidden[:,-1,:].reshape(B1,M1,-1)
        logits = self.actor(actor_hidden).reshape(B,M,3,5)
        if action is None:
            if self.test==True:
                action=torch.argmax(logits,dim=-1)
            elif self.test==False:
                action = Categorical(logits = logits).sample()
        logprob = Categorical(logits = logits).log_prob(action)
        entropy = Categorical(logits = logits).entropy()
        if squeeze==True:
            logprob=logprob.squeeze(1)
            entropy=entropy.squeeze(1)
            critic_hidden=critic_hidden.squeeze(1)
        return action, logprob.sum(-1), entropy.sum(-1), self.critic(critic_hidden)


class c_Agent_lstm(nn.Module):
    def __init__(self, envs,hidden_size=64):
        super(c_Agent_lstm, self).__init__()
        self.test=False
        self.critic_GRU = nn.Sequential(
            layer_init(nn.Linear(int((envs.single_observation_space.shape[2]-11)*envs.single_observation_space.shape[1]), hidden_size)),
            nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True),
        )
        self.critic=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std = 1.0))
        self.actor_GRU = nn.Sequential(
            layer_init(nn.Linear(int((envs.single_observation_space.shape[2]-11)*envs.single_observation_space.shape[1]), hidden_size)),
            nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True),
        )
        self.actor=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size,int(envs.single_observation_space.shape[1]*envs.single_action_space[0].n)), std = 0.01))
    def forward(self, x_actor,x_critic,action=None):
        squeeze=False
        if len(x_actor.shape)==3:
            squeeze=True
            x_actor=x_actor.unsqueeze(2)
            x_critic=x_critic.unsqueeze(2)
            action=action.unsqueeze(1)
        B,L,M,H=x_actor.shape
        B1, L1, M1, H1 = x_critic.shape
        x_actor=x_actor.reshape(B,L,M*H)
        x_critic=x_critic.reshape(B1,L1,M1*H1)
        actor_hidden,_=self.actor_GRU(x_actor)
        actor_hidden=actor_hidden[:,-1,:]
        critic_hidden,_=self.critic_GRU(x_critic)
        critic_hidden=critic_hidden[:,-1,:].reshape(B1,-1)
        logits = self.actor(actor_hidden).reshape(B,M,3,5)
        if action is None:
            if self.test==True:
                action=torch.argmax(logits,dim=-1)
            elif self.test==False:
                action = Categorical(logits = logits).sample()
        logprob = Categorical(logits = logits).log_prob(action)
        entropy = Categorical(logits = logits).entropy()
        if squeeze==True:
            logprob=logprob.squeeze(1)
            entropy=entropy.squeeze(1)
            critic_hidden=critic_hidden.squeeze(1)
        return action, logprob.sum(-1), entropy.sum(-1), self.critic(critic_hidden).unsqueeze(dim=1).repeat(1,action.shape[1],1)

class GAT(nn.Module):
    def __init__(self,Gmat,Attention_in_features,Attention_hidden_features,n_heads):
        super(GAT,self).__init__()

        self.Qweight=nn.Parameter(torch.rand(Attention_hidden_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)
        self.Kweight=nn.Parameter(torch.rand(Attention_hidden_features,Attention_hidden_features)*((4/Attention_in_features)**0.5)-(1/Attention_in_features)**0.5)

        self.linear=nn.Linear(Attention_in_features,Attention_hidden_features)
        self.linear2=nn.Linear(2*Attention_hidden_features,Attention_hidden_features)
        self.Gmat=Gmat
        self.n_heads=n_heads

#        self.critic=critic
    def forward(self,s):

        s=self.linear(s)
        B,L,_=s.shape
        H=self.n_heads
        
        q=torch.einsum('bjk,km->bjm',s,self.Qweight).view(B, L, H, -1)
        Gmat=self.Gmat.to(s.device).to(s.dtype)
        
        k=torch.einsum('bjk,km->bjm',s,self.Kweight).view(B, L, H, -1)
        s=s.view(B, L, H, -1)
        att=torch.einsum("blhe,bshe->bhls", q, k)/(128**0.5)
#        att=torch.bmm(q,k)
#        att = F.softmax(att, dim=-1)
#        att=torch.bmm(q,k)
#        att=torch.softmax(att*Gmat,dim=2)

#        att=torch.bmm(q,k)/(128**0.5)

        att = torch.where(Gmat > 0, att,-9e15*torch.ones_like(att))
        att = F.softmax(att, dim=-1)

        s_out=torch.einsum('bshe,bhls->blhe',s,att).view(B,L,-1)
        s=s.view(B,L,-1)
#        att=att/(torch.sum(att,dim=2,keepdim=True)+1e-6)
#        s_out=torch.einsum('bjk,bjm->bmk',s,att)
#        s_out=s_out+s
        s_out=self.linear2(torch.cat((s,s_out),dim=-1))
        return s_out





class GCN(nn.Module):
    def __init__(self, A, dim_in, dim_hidden):
        super(GCN, self).__init__()

        A = A + torch.eye(A.shape[0])  # A = A+I
        D = torch.diag(torch.pow(A.sum(dim=1), -0.5))  # D = D^-1/2
        self.A = D @ A @ D  # \hat A

        self.input_fc = nn.Linear(dim_in, dim_hidden, bias=False)
	
        self.fc1 = nn.Linear(dim_hidden, dim_hidden, bias=False)
#        self.fc2 = nn.Linear(dim_hidden, dim_hidden, bias=False)
        self.fc3 = nn.Linear( dim_hidden*2, dim_hidden, bias=False)

    def forward(self, X):
    
        X = self.input_fc(X)
        self.A=self.A.to(X.device)
        X_=X
        X = self.fc1(self.A @ X)
#        X = F.relu(self.fc2(self.A @ X))

        out=self.fc3(torch.cat((X,X_),dim=-1))

        
        return out



class Agent_gcnlstm(nn.Module):
    def __init__(self, envs,hidden_size=64):
        super(Agent_gcnlstm, self).__init__()
        self.test=False

        self.A = torch.load('/EPC_RF/env/hzone_contact_num.pt')
        self.A_B = torch.load('./env/jd_hzone_contact.pt')
        self.linear_actor=nn.Linear( envs.single_observation_space.shape[2]-11, hidden_size)
#        self.linear_critic=nn.Linear( envs.single_observation_space.shape[2]-10, hidden_size)


        self.GCN1 = GAT(self.A, envs.single_observation_space.shape[2]-11, hidden_size,4)
        self.GCN2 = GAT(self.A, envs.single_observation_space.shape[2]-11, hidden_size,4)
        self.critic_GRU =nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True)

#        self.linear_actor=nn.Linear(2*hidden_size, hidden_size)
        self.linear_critic=nn.Linear(3*hidden_size, hidden_size)
#
#        self.input_layer=nn.Linear(envs.single_observation_space.shape[2]-10, hidden_size)

        self.critic=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, 1), std = 1.0))
#        self.actor_GCN =GATmodel(envs.single_observation_space.shape[2]-10, hidden_size)
        self.actor_GRU = nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers = 1,batch_first=True)

        self.actor=nn.Sequential(
            layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_size, envs.single_action_space[0].n), std = 0.01))

        
    def forward(self, x_actor,x_critic,action=None):
        squeeze=False
#        torch.manual_seed(42)
#        self.edge_index=self.edge_index.to(x_actor.device)
#        gm=self.edge_index.to(x_actor.device)
        if len(x_actor.shape)==3:
            squeeze=True
            x_actor=x_actor.unsqueeze(2)
            x_critic=x_critic.unsqueeze(2)
            action=action.unsqueeze(1)
        B,L,M,H=x_actor.shape
        B1, L1, M1, H1 = x_critic.shape
        # x_actor=x_actor.permute(1,0,2,3).reshape(L,B*M,H).permute(1,0,2)
        # x_critic=x_critic.permute(1,0,2,3).reshape(L1,B1*M1,H1).permute(1,0,2)
        x_actor = x_actor.permute(1, 0, 2, 3).reshape(L*B,M, H)
        

#        x_actor=self.input_layer(x_actor)
        x_critic = x_actor
        
        x_actor= self.linear_actor(x_actor)
        x_critic = self.GCN2(x_critic)
#
#        x_actor= self.linear_actor(x_actor)
#        x_critic = self.linear_critic(x_critic)

        
#        self.A_B=self.A_B.to(x_critic.device).to(x_critic.dtype)
#        
#        x_interm=torch.matmul(self.A_B, x_critic_local) / self.A_B.sum(dim=1, keepdim=True)
#        x_interm=x_interm[:,torch.where(self.A_B==1)[0],:]
#        x_global = torch.mean( x_critic_local, dim=1,keepdim = True).repeat(1,M,1)
#        x_critic=self.linear_critic(torch.cat(( x_critic_local,x_interm,x_global),dim=2))




        # 创建Data对象列表
        # x_actor_list = [Data(x = x_actor[i], edge_index = self.edge_index) for i in range( x_actor.shape[0])]
        # batch_x_actor = Batch.from_data_list(x_actor_list)
        #
        # x_critic_list = [Data(x = x_critic[i], edge_index = self.edge_index) for i in range( x_critic.shape[0])]
        #
        # # 使用Batch来合并这些Data对象
        # batch_x_critic = Batch.from_data_list(x_critic_list)

#        x_actor = self.GCN(x_actor, gm)
#        x_critic = self.GCN(x_critic, gm)

        x_actor = x_actor.reshape(L, B*M, -1).permute(1, 0, 2)
        x_critic = x_critic.reshape(L1, B1*M1, -1).permute(1, 0, 2)
        actor_hidden,_=self.actor_GRU(x_actor)
        actor_hidden=actor_hidden[:,-1,:]
        critic_hidden,_=self.critic_GRU(x_critic)
        critic_hidden=critic_hidden[:,-1,:].reshape(B1,M1,-1)
        logits = self.actor(actor_hidden).reshape(B,M,3,5)
        if action is None:
            if self.test==True:
                action=torch.argmax(logits,dim=-1)
            elif self.test==False:
                action = Categorical(logits = logits).sample()
        logprob = Categorical(logits = logits).log_prob(action)
        entropy = Categorical(logits = logits).entropy()
        if squeeze==True:
            logprob=logprob.squeeze(1)
            entropy=entropy.squeeze(1)
            critic_hidden=critic_hidden.squeeze(1)
        return action, logprob.sum(-1), entropy.sum(-1), self.critic(critic_hidden)

        # elif len(x_actor.shape)==3:
        #     logits = self.actor(x_actor).reshape(x_actor.shape[0],3,5)
        #     if action is None:
        #         action = Categorical(logits = logits).sample()
        #     logprob = Categorical(logits = logits).log_prob(action).reshape(x_actor.shape[0],-1)
        #     entropy = Categorical(logits = logits).entropy().reshape(M,-1)
        #     return action, logprob.sum(-1), entropy.sum(-1),self.critic(critic_hidden)

    # def get_action_and_value(self, x, action = None):
    #     hidden = self.network(x)
    #     logits = self.actor(hidden)
    #     split_logits = torch.split(logits, self.nvec.tolist(), dim = 1)
    #     multi_categoricals = [Categorical(logits = logits) for logits in split_logits]
    #     if action is None:
    #         action = torch.stack([categorical.sample() for categorical in multi_categoricals])
    #     logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
    #     entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
    #     return action.T, logprob.sum(0), entropy.sum(0), self.critic(hidden)
# def forward(self, x_actor,x_critic,action=None):
    #     action_mean = self.actor_mean(x_actor)
    #     action_logstd = self.actor_logstd.expand_as(action_mean)
    #     action_std = torch.exp(action_logstd)
    #     probs = Normal(action_mean, action_std)
    #     if action is None:
    #         action = probs.sample()
    #         # action[:,:,0]=(torch.exp(-1*action[:,:,0])+torch.exp(-1*action[:,:,1])+torch.exp(-1*action[:,:,2]))/(torch.exp(-1*action[:,:,0])+torch.exp(-1*action[:,:,1])+torch.exp(-1*action[:,:,2])+torch.exp(-1*action[:,:,3]))
    #         # action[:,:,1]=(torch.exp(-1*action[:,:,0])+torch.exp(-1*action[:,:,1]))/(torch.exp(-1*action[:,:,0])+torch.exp(-1*action[:,:,1])+torch.exp(-1*action[:,:,2])+torch.exp(-1*action[:,:,3]))
    #         # action[:,:,2]=(torch.exp(-1*action[:,:,0]))/(torch.exp(-1*action[:,:,0])+torch.exp(-1*action[:,:,1])+torch.exp(-1*action[:,:,2])+torch.exp(-1*action[:,:,3]))
    #         # action[:,:,3]=1/(1+torch.exp(-1*action[:,:,3]))
    #         # action[:, :, 0]=1/(1+torch.exp(-2*action[:, :, 0]))
    #         # action[:, :, 1]=0.2/(1+torch.exp(-2*action[:, :, 1]))
    #         # action[:, :, 2]=0.04/(1+torch.exp(-2*action[:, :, 2]))
    #     return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x_critic)

# class Agent(nn.Module):
#     def __init__(self, envs,hidden_size=64):
#         super(Agent, self).__init__()
#
#
#         self.critic = nn.Sequential(
#             layer_init(nn.Linear(envs.single_observation_space.shape[1], hidden_size)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, hidden_size)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, 1), std = 1.0),
#         )
#         self.actor_mean = nn.Sequential(
#             layer_init(nn.Linear(int(envs.single_observation_space.shape[1]-8), hidden_size)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, hidden_size)),
#             nn.Tanh(),
#             layer_init(nn.Linear(hidden_size, envs.single_action_space.shape[1]), std = 0.01),
#         )
#         # self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape[1])))
#
#     def forward(self, x_actor,x_critic,action=None,new_action_std=None):
#         action_mean = self.actor_mean(x_actor)
#         action_std = torch.full((action_mean.shape), new_action_std).to(action_mean.device)
#         # action_std = new_action_std.expand_as(action_mean)
#         probs = Normal(action_mean, action_std)
#         if action is None:
#             action = probs.sample()
#         return action, probs.log_prob(action).sum(-1), probs.entropy().sum(-1), self.critic(x_critic)
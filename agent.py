import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from model_pn import Actor, Critic, Critic_2
from replay_buffer import ReplayBuffer

from scipy import *

from scipy.optimize import minimize

from torch.autograd import Variable

class Agent:
    def __init__(self, args):
        self.args = args
        # self.buffer_size = 20000
        # self.batch_size = 25
        self.replay_buffer = ReplayBuffer(self.args)
        self.antenna_number = self.args.user_antennas
        self.user_number = self.args.user_numbers
        self.bs_antenna_number = self.args.bs_antennas
        self.device = 'cuda' if self.args.cuda else 'cpu'
        # self.device = 'cpu'
        self.writer = self.args.writer
        self.policy_net = Actor(self.args).to(self.device)
        self.critic_net = Critic(self.args).to(self.device)
        self.critic_net_2 = Critic_2(self.args).to(self.device)
        # 定义两个active网络学习率的decay系数
        self.learning_rate_policy_net = self.args.actor_lr
        self.learning_rate_critic_net = self.args.critic_lr
        self.decay_rate_policy_net = self.args.actor_lr_decay
        self.decay_rate_critic_net = self.args.critic_lr_decay
        
        # discount ratio
        self.gamma = self.args.gamma
        # GAE factor
        self.GAE_factor = self.args.GAE_factor
        # 定义loss function
        self.loss_function = nn.MSELoss()
        # 定义优化器
        self.optimizer_policy = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate_policy_net)
        self.optimizer_critic = optim.Adam(self.critic_net.parameters(), lr=self.learning_rate_critic_net)
        self.optimizer_critic_2 = optim.Adam(self.critic_net_2.parameters(), lr=self.learning_rate_critic_net)
        
        self.update_value_net_count = 0
        self.update_value_net_2_count = 0
        self.update_policy_net_count = 0

        self.weight_vector = [1,0]
        np.save('./Exp/new_init_weight',self.weight_vector)

    def Pick_action(self, channel_matrix, user_reward,noise,sample=True):  #Actor
        # 将channel_matrix, user_reward转换成tensor
        channel_matrix = torch.FloatTensor(channel_matrix).to(self.device)
        user_reward = torch.FloatTensor(user_reward).to(self.device).unsqueeze(-1)
        noise=torch.FloatTensor(noise).to(self.device)
        prob_value,schedule_result = self.policy_net(channel_matrix, user_reward,noise,sample)
        return schedule_result, prob_value

    def Update_value_net(self, target_value, channel_matrix, user_reward):   #update critic

        for i in range(1):
            self.update_value_net_count += 1
            self.optimizer_critic.zero_grad()
            feature_f = self.critic_net_2(channel_matrix, user_reward)[1].detach()
            approximate_value = self.critic_net(channel_matrix, user_reward,feature_f)[0]
            loss = self.loss_function(approximate_value, target_value)
            loss.backward()
            params_e = list(self.critic_net.parameters())
            grad_c = [p.grad.data.cpu().numpy() for p in params_e]
            self.optimizer_critic.step()
            self.writer.add_scalar('Loss/new_critic_loss', loss.item(), self.update_value_net_count)
        return grad_c

    def Update_value_net_2(self, target_value, channel_matrix, user_fairness_reward):   #update critic

        for i in range(1):
            self.update_value_net_2_count += 1
            self.optimizer_critic_2.zero_grad()
            approximate_value = self.critic_net_2(channel_matrix, user_fairness_reward)[0]
            loss = self.loss_function(approximate_value, target_value)
            loss.backward()
            params_f = list(self.critic_net_2.parameters())
            grad_f = [p.grad.data.cpu().numpy() for p in params_f]
            self.optimizer_critic_2.step()
            self.writer.add_scalar('Loss/new_critic_loss', loss.item(), self.update_value_net_2_count)
        return grad_f

    def Training(self):
        # 这个地方将进行两个网络的训练
        Trajectories = self.replay_buffer.sample()
        channel_matrix = Trajectories['Channel']
        instant_capacity_reward = Trajectories['instant_capacity_reward']
        instant_fairness_reward = Trajectories['instant_fairness_reward']
        user_fairness_reward = Trajectories['Average_fairness_reward']
        mask = Trajectories['terminate']
        probs = Trajectories['prob']
        noise = Trajectories['noise']

        channel_matrix = torch.FloatTensor(channel_matrix).to(self.device).reshape(-1, self.args.channel_dim1, self.args.channel_dim2)
        user_fairness_reward = torch.FloatTensor(user_fairness_reward).to(self.device).reshape(-1, self.args.channel_dim1,1)
        instant_capacity_reward = torch.FloatTensor(instant_capacity_reward).to(self.device).reshape(-1)
        instant_fairness_reward = torch.FloatTensor(instant_fairness_reward).to(self.device).reshape(-1)
        mask = torch.FloatTensor(mask).to(self.device).reshape(-1)
        Prob = torch.stack([torch.stack(probs[i], 0) for i in range(self.args.episodes)], 0).reshape(-1)
        noise = torch.FloatTensor(noise).to(self.device).squeeze()


        values_f, feature_f = self.critic_net_2(channel_matrix, user_fairness_reward)
        values_f = values_f.detach()
        feature_f = feature_f.detach()
        returns_f = torch.zeros(channel_matrix.shape[0],1).to(self.device)
        deltas_f = torch.Tensor(channel_matrix.shape[0],1).to(self.device)
        advantages_f = torch.Tensor(channel_matrix.shape[0],1).to(self.device)

        values_c = self.critic_net(channel_matrix, user_fairness_reward,feature_f)[0].detach()  # critic
        returns_c = torch.zeros(channel_matrix.shape[0], 1).to(self.device)  # target
        deltas_c = torch.Tensor(channel_matrix.shape[0], 1).to(self.device)  # target - critic
        advantages_c = torch.Tensor(channel_matrix.shape[0], 1).to(self.device)

        prev_return_c = 0
        prev_value_c = 0
        prev_advantage_c = 0

        prev_return_f = 0
        prev_value_f = 0     
        prev_advantage_f = 0

        for i in reversed(range(instant_capacity_reward.shape[0])):
            returns_c[i] = instant_capacity_reward[i] + self.gamma * prev_return_c * mask[i]
            deltas_c[i] = instant_capacity_reward[i] + self.gamma * prev_value_c * mask[i] - values_c.data[i]
            advantages_c[i] = deltas_c[i] + self.gamma * self.GAE_factor * prev_advantage_c * mask[i]

            returns_f[i] = instant_fairness_reward[i] + self.gamma * prev_return_f * mask[i]
            deltas_f[i] = instant_fairness_reward[i] + self.gamma * prev_value_f * mask[i] - values_f.data[i]
            advantages_f[i] = deltas_f[i] + self.gamma * self.GAE_factor * prev_advantage_f * mask[i]

            prev_return_c = returns_c[i, 0]
            prev_value_c = values_c.data[i, 0]
            prev_advantage_c = advantages_c[i, 0]

            prev_return_f = returns_f[i, 0]
            prev_value_f = values_f.data[i, 0]
            prev_advantage_f = advantages_f[i, 0]

        grad_c = self.Update_value_net(returns_c, channel_matrix, user_fairness_reward)
        grad_f = self.Update_value_net_2(returns_f, channel_matrix, user_fairness_reward)

        ######################################################. update weight_vector

        advantages_c = (advantages_c - torch.mean(advantages_c)) / torch.std(advantages_c)
        advantages_f = (advantages_f - torch.mean(advantages_f)) / torch.std(advantages_f)
        def func(weight_vector = [0.5,0.5]):
            s = 0
            for i in range(len(grad_f)):
                s += np.sum(np.square(grad_c[i]*weight_vector[0]+grad_f[i]*weight_vector[1]))
            return s
 
        # cons = ({'type': 'eq','fun': lambda x: x[0] + x[1] - 1 },
        #             {'type': 'ineq', 'fun': lambda x: x[0] - 0},{'type': 'ineq', 'fun': lambda x: x[1] - 0})
        # res = minimize(func, self.weight_vector, constraints=cons, method='SLSQP', options={'disp': True})
        # self.weight_vector = res.x
        w1 = self.weight_vector[0]
        w2 = self.weight_vector[1]
        self.update_policy_net_count += 1
        advantages=torch.cat((advantages_c,advantages_f),1)
        policy_net_loss = - torch.mean(Prob* (torch.sum(torch.multiply(advantages,noise[0]),1)))
        # advantages = w1*advantages_c + w2*advantages_f
        # policy_net_loss = - torch.mean(Prob * advantages.squeeze())
        self.optimizer_policy.zero_grad()
        policy_net_loss.backward()
        self.optimizer_policy.step()
        # self.writer.add_scalar('Loss/actor_loss', torch.mean(torch.exp(Prob) * advantages.squeeze()), self.update_policy_net_count)
        self.learning_rate_critic_net = (1-self.decay_rate_critic_net) * self.learning_rate_critic_net
        self.learning_rate_policy_net = (1-self.decay_rate_policy_net) * self.learning_rate_policy_net
        self.replay_buffer.reset_buffer()
        return w1,w2
    def Store_transition(self, batch):
        self.replay_buffer.store_episode(batch)
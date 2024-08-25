import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np

class Actor(nn.Module):
    def __init__(self, args):
    # def __init__(self, flatten_dim , high_space_dim, weight_dim=32, hidden=64):
        super(Actor, self).__init__()
        self.args = args
        self.eval=False
        self.noise_dim = self.args.noise_dim
        flatten_dim=120+self.noise_dim
        output_dim=self.args.user_antennas * self.args.user_numbers
        self.conv1=nn.Conv2d(1,1,(1,5),(1,1))
        self.conv2=nn.Conv2d(1,1,(1,3),(1,2))
        self.embedding_noise = nn.Linear(2, self.noise_dim)
        self.flatten=nn.Flatten(start_dim=0)
        self.linear1=nn.Linear(flatten_dim,128)
        self.linear2=nn.Linear(128,32)
        self.linear3=nn.Linear(32,output_dim)
        self.tanh=nn.Tanh()
        self.sigmoid=nn.Sigmoid()
        self.epsilon = self.args.epsilon
        self.soft=0.01

    def forward(self, channel_matrix, reward_array,noise):
        # 默认输入进来的信道矩阵是batch * 20 * 32的
        input_data = 1e5 * channel_matrix
        latent=torch.cat((input_data,reward_array),-1)
        latent=self.conv1(latent.unsqueeze(0).unsqueeze(0))
        latent=self.tanh(latent)
        latent=self.conv2(latent)
        latent=self.tanh(latent)
        
        embedding_noise = self.embedding_noise(noise)
        embedding_noise = self.tanh(embedding_noise)
        # latent=self.conv3(latent)

        latent=self.flatten(latent)
        latent=torch.cat((latent,embedding_noise),0)
        # latent=self.tanh(latent)
        latent=self.linear1(latent)
        latent=self.tanh(latent)
        latent=self.linear2(latent)
        latent=self.tanh(latent)
        latent=self.linear3(latent)
        # latent=self.tanh(latent)
        # latent=self.linear4(latent)
        output=(self.sigmoid(latent)-0.5)*0.99+0.5
        if torch.rand(1)<self.epsilon and self.eval==False:
            prob_matrix=torch.stack((1-output,output),1)
            # batch_size=prob_matrix.shape[0]
            # user_num=prob_matrix.shape[1]
            # sample_matrix=torch.reshape(prob_matrix,(-1,2))
            result=torch.multinomial(prob_matrix,1).reshape(-1)
            prob_=prob_matrix[range(result.shape[0]),result]
        else:
            result=torch.where(output>=0.5,1,0)
            prob_=torch.where(result==1,output,1-output)
        log_prob=torch.log(torch.prod(prob_))
        schedule_result=torch.where(result==1)[0]
        print(schedule_result)
        print(log_prob)
        return log_prob, schedule_result

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.device = 'cuda' if self.args.cuda else 'cpu'
        self.embedding_dim = self.args.embedding_dim
        self.flatten_dim = self.args.flatten_dim
        self.embedding_layer = nn.Linear(self.flatten_dim, self.embedding_dim)
        self.fc_layer_number = self.args.fc_layer_number
        self.hidden_dim = self.args.hidden_dim
        input_dim = self.embedding_dim + 1
        self.fc_net = []
        for layer in range(self.fc_layer_number):
            self.fc_net.append(nn.Linear(input_dim, self.hidden_dim[layer]))
            input_dim = self.hidden_dim[layer]
        self.fc_net=nn.Sequential(*self.fc_net)
        self.flatten = nn.Flatten()
        self.ouput_layer = nn.Linear(input_dim*self.args.user_antennas * self.args.user_numbers, 1)

    def forward(self, channel_matrix, reward_array):
        input_data = 1e6 * channel_matrix
        embedding_data = self.embedding_layer(input_data)
        fc_data = torch.cat((embedding_data, reward_array), 2)
        for layer in range(self.fc_layer_number):
            fc_data = self.fc_net[layer](fc_data)
        flatten_vector = self.flatten(fc_data)
        value = self.ouput_layer(flatten_vector)
        return value


class Critic_2(nn.Module):
    
    def __init__(self, args):
        super(Critic_2, self).__init__()
        self.args = args
        self.device = 'cuda' if self.args.cuda else 'cpu'
        self.embedding_dim = self.args.embedding_dim
        self.flatten_dim = self.args.flatten_dim
        self.embedding_layer = nn.Linear(self.flatten_dim, self.embedding_dim)
        self.fc_layer_number = self.args.fc_layer_number
        self.hidden_dim = self.args.hidden_dim
        input_dim = self.embedding_dim + 1
        self.fc_net = []
        for layer in range(self.fc_layer_number):
            self.fc_net.append(nn.Linear(input_dim, self.hidden_dim[layer]))
            input_dim = self.hidden_dim[layer]
        self.fc_net=nn.Sequential(*self.fc_net)
        self.flatten = nn.Flatten()
        self.ouput_layer = nn.Linear(input_dim*self.args.user_antennas * self.args.user_numbers, 1)

    def forward(self, channel_matrix, reward_array):
        input_data = 1e6 * channel_matrix
        embedding_data = self.embedding_layer(input_data)
        fc_data = torch.cat((embedding_data, reward_array), 2)
        for layer in range(self.fc_layer_number):
            fc_data = self.fc_net[layer](fc_data)
        flatten_vector = self.flatten(fc_data)
        value = self.ouput_layer(flatten_vector)
        return value


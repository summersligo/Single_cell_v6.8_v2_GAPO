# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import copy
from tqdm import tqdm
from scipy.io import loadmat





class Environment:
    def __init__(self, args):
        self.args = args
        self.user_num = self.args.user_numbers
        self.antenna_number = self.args.user_antennas
        self.bs_antenna_number = self.args.bs_antennas
        self.total_antennas = self.user_num * self.antenna_number
        self.tau = self.args.tau
        self.t_c = self.args.t_c
        self.TTI_length = self.args.TTI_length
        self.transmit_power = self.args.transmit_power
        self.noise_power = self.args.noise_power
        self.data_folder = self.args.data_folder
        self.training_set_length = self.args.episodes
        self.read_training_data()
        self.power = self.transmit_power / self.bs_antenna_number
        self.basis_num= self.user_num*self.antenna_number*self.TTI_length

    def read_training_data(self):
        # variable file_index is the index of current file
        self.file_index = 0
        self.training_set = []
        file_list = sorted(os.listdir(self.data_folder))
        # Read chanel data set
        for i in range(self.training_set_length):
            self.training_set.append(self.data_folder + "/" + file_list[i])
    

    def Reset(self):
        # TTI_count represent the time index of current channel file
        self.TTI_count = 50
        # Read specific channel data based on file_index 
        self.episode_data = np.load(self.training_set[self.file_index])
        self.file_index += 1
        self.Calculate_average_fairness_reward()
        self.Read_TTI_data()
        

    def Read_TTI_data(self):
        # Since multi-agent environment, we need return observations and global state
        self.TTI_data = self.episode_data[:,:,self.TTI_count]
        self.TTI_data = np.concatenate([self.TTI_data[:,:self.bs_antenna_number],self.TTI_data[:,16:16+self.bs_antenna_number]],1)
        self.TTI_count += 1

    def Calculate_average_fairness_reward(self,  instant_fairness_reward=None):
        if not instant_fairness_reward:
            self.average_fairness_reward = np.ones(self.total_antennas)*1e-6
        else:
            self.average_fairness_reward = self.average_fairness_reward* self.t_c + (1-self.t_c) * np.array(instant_fairness_reward)


    def Calculate_precoding_matrix(self):
        # This function is used to calculate precoding matrix. If action of current cell is satisfied stream schedule rule
        # then this cell will have precoding matrix, otherwise, the precoding matrix is setted as None
        if self.is_reasonable:
            # Take out correspond channel matrix
            channel_matrix = self.select_data
            # Calculate pseudo_inverse (H^HH+1e-6I)^-1H
            pseudo_inverse = np.linalg.pinv(channel_matrix)
            # Nomalize of pseudo_inverse matrix
            F_norm = np.sum(np.abs(pseudo_inverse)**2)
            self.precoding_matrix = pseudo_inverse/np.sqrt(F_norm)
            # precoding_matrix.append(pseudo_inverse)
        else:
            self.precoding_matrix = None

    def Select_channel_data(self, action):
        if self.is_reasonable:
            channel_matrix = self.TTI_data
            rebuild_matrix = channel_matrix[:, 0:self.bs_antenna_number] + 1j*channel_matrix[:, self.bs_antenna_number:]

            #0:8; 16:32
            self.select_data = rebuild_matrix[np.array(action.cpu().detach()), :]
        else:
            self.select_data = None
    
    def Action_reasonable(self, action):
        # Define a list, which is used to decide whether is reasonable of arbitrary cell action
        if len(action) > self.args.max_stream or len(action) == 0:
            self.is_reasonable = False
        else:
            self.is_reasonable = True
    
    
    def Calculate_user_sum_rate(self, action):
        users_sum_rate = []
        user_count = 0
        schedule_user_number = len(action)
        schedule_user_set = [i for i in range(schedule_user_number)]
        if self.is_reasonable:
            for user in range(self.total_antennas):
                # traverse all actions, if action has selected by the policy net, SINR will be calculated, otherwise, directly add zero
                if user in action:
                    antenna_channel = self.select_data[user_count, :]
                    # 计算分子和分母部分, 分母分成两个部分，一个是当前小区之内的干扰，另外一个是相邻小区的干扰
                    Molecular = self.power * np.abs(np.sum(antenna_channel * self.precoding_matrix[:,user_count])) **2
                    Intra_interference_user = schedule_user_set.copy()
                    Intra_interference_user.remove(user_count)
                    if len(Intra_interference_user) == 0:
                        Intra_interference_value = 0
                    else:
                        Intra_precoding_matrix = self.precoding_matrix[:, np.array(Intra_interference_user)]
                        # 如果长度是大于1， 则显然其需要先计算一个向量
                        Intra_interference_value = self.power * np.sum(np.abs(antenna_channel[np.newaxis, :].dot(Intra_precoding_matrix)) **2)
                    Dominator = self.noise_power + Intra_interference_value 
                    user_sum_rate = np.log2(1+Molecular/Dominator)
                    users_sum_rate.append(user_sum_rate)
                    user_count += 1
                else:
                    users_sum_rate.append(0)
        else:
            # 这个表示的当前小区没有数据进行发送，因此也就直接将所有用户的instant reward设置为0
            for user in range(self.total_antennas):
                users_sum_rate.append(0)
        return users_sum_rate

    def Calculate_reward(self, action):
        # Traversal all cell, and calculate the instant rate of individual user
        # First, calculate precodin matrix
        self.Calculate_precoding_matrix()
        users_capacity_reward = self.Calculate_user_sum_rate(action) 
        users_fairness_reward = users_capacity_reward/self.average_fairness_reward
        self.Calculate_average_fairness_reward(users_capacity_reward)

        return np.sum(users_capacity_reward), np.sum(users_fairness_reward) if np.sum(users_fairness_reward)<40 else 30

    def Step(self, action):
        terminated = False
        self.Action_reasonable(action)
        self.Select_channel_data(action)
        capacity_instant_reward, fairness_instant_reward = self.Calculate_reward(action)
        if self.TTI_count == self.TTI_length:
            terminated = True
        else:
            self.Read_TTI_data()
        return capacity_instant_reward, fairness_instant_reward, terminated

    def get_state(self):
        # apart channel matrix and average reward
        channel = copy.deepcopy(self.TTI_data)
        average_fairness_reward = copy.deepcopy(self.average_fairness_reward)
        return channel, average_fairness_reward

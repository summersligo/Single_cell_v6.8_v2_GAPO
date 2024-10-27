import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from arguments import get_A2C_args
from Env import Environment
from agent import Agent

import matplotlib.colors as mcolors
colors=list(mcolors.TABLEAU_COLORS.keys()) #颜色变化

import random
seed=11129
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class Project:
    def __init__(self):
        self.args = get_A2C_args()
        # create three folders which is used to store model, figure and tensorboard data
        self.create_folder(self.args.result_folder)
        self.create_folder(self.args.model_folder)
        self.create_folder(self.args.vision_folder)
        # create a summary writer, which is used to record the loss value of individual agent
        self.args.writer = SummaryWriter(self.args.vision_folder)
        self.agent = Agent(self.args)
        # define environment
        self.env = Environment(self.args)
        self.save_interval=100
        self.history='Model'

    def create_folder(self, folder_name):
        # create a folder, if folder exists, delete it first.
        if os.path.exists(folder_name):
            pass
            # shutil.rmtree(folder_name)
            # os.mkdir(folder_name)
        else:
            os.mkdir(folder_name)
        pass

    
    def Simulation(self):
        epoch_capacity_cumulative_reward =[] 
        epoch_fairness_cumulative_reward =[]
        w1_collector=[]
        w2_collector=[]
        # optimal_reward = 0
        # noise=
        # index=10900
        # self.load_model(index)
        for ep in tqdm(range(1,self.args.epoches+1)):
            episode_capacity_reward = []
            episode_fairness_reward = []
            episode_capacity_reward_collector = []
            episode_fairness_reward_collector = []
            self.env.read_training_data()
            for _ in range(self.args.episodes):
                # collect experience
                # for i in range(self.args.noise.shape[0]):
                cumulative_capacity_reward, cumulative_fairness_reward, actions = self.generate_episode(self.args.noise[(ep-1)%21])
            cumulative_capacity_reward, cumulative_fairness_reward, actions = self.test_episode(self.args.noise[0])
            episode_fairness_reward.append(cumulative_fairness_reward)
            cumulative_capacity_reward, cumulative_fairness_reward, actions = self.test_episode(self.args.noise[-1])            
            episode_capacity_reward.append(cumulative_capacity_reward)

            average_episode_capacity_reward = np.sum(episode_capacity_reward)
            average_episode_fairness_reward = np.sum(episode_fairness_reward)

            print("Epoch: {}, Current epoch average cumulative capacity reward is: {};  fairness_reward is {}".format(ep, average_episode_capacity_reward,average_episode_fairness_reward))
            epoch_capacity_cumulative_reward.append(average_episode_capacity_reward)
            epoch_fairness_cumulative_reward.append(average_episode_fairness_reward)
            w1,w2=self.agent.Training()
            w1_collector.append(w1)
            w2_collector.append(w2)
            if ep % self.save_interval==0:
                self.agent.policy_net.eval=True
                w1_test=torch.arange(0,1.05,0.05)
                test_noise=torch.stack((w1_test,1-w1_test),1)
                cumulative_capacity_reward=[]
                cumulative_fairness_reward=[]
                # print(test_noise)
                for i in range(test_noise.shape[0]):
                    self.agent.replay_buffer.index=0
                    capacity,fairness,actions=self.test_episode(test_noise[i])
                    cumulative_capacity_reward.append(capacity)
                    cumulative_fairness_reward.append(fairness)
                # cumulative_capacity_reward, cumulative_fairness_reward = 
                cumulative_capacity_reward=np.array(cumulative_capacity_reward)
                cumulative_fairness_reward=np.array(cumulative_fairness_reward)
                np.save('./Exp/new_capacity_%06d'%ep,cumulative_capacity_reward)
                np.save('./Exp/new_fairness_%06d'%ep,cumulative_fairness_reward)
                self.agent.policy_net.eval=False
                plt.figure(figsize=(15,10))
                for i in range(cumulative_capacity_reward.shape[0]):
                    plt.scatter(cumulative_capacity_reward[i],cumulative_fairness_reward[i],color=mcolors.TABLEAU_COLORS[colors[i%10]])
                    plt.annotate("(%.2f,%.2f)"%(test_noise[i][0],test_noise[i][1]),xy=(cumulative_capacity_reward[i],cumulative_fairness_reward[i]))
                plt.xlabel("capacity")
                plt.ylabel("fairness")
                plt.savefig('./Figure/new_epoch_%d_MO_reward.png'%ep)
                plt.close()
                self.save_model(ep)
                self.plot_figure(ep,np.array(epoch_capacity_cumulative_reward), "capacity")
                self.plot_figure(ep,np.array(epoch_fairness_cumulative_reward), "fairness")
                self.agent.replay_buffer.index=0
            if self.args.epsilon <= self.args.min_epsilon:
                self.args.epsilon = self.args.min_epsilon
            else:
                self.args.epsilon = self.args.epsilon - self.args.epsilon_decay
            print(self.args.epsilon)


    def testUnseen(self):
        self.agent.policy_net.load_state_dict(torch.load('./Model/4500_Agent_policy_net_new.pkl'))
        epoch_capacity_cumulative_reward =[] 
        epoch_fairness_cumulative_reward =[]
        w1_collector=[]
        w2_collector=[]
        # optimal_reward = 0
        # noise=
        # index=10900
        # self.load_model(index)
        self.agent.policy_net.eval=True
        w1_test=torch.arange(0,1.0,0.01)
        test_noise=torch.stack((w1_test,1-w1_test),1)
        cumulative_capacity_reward=[]
        cumulative_fairness_reward=[]
        test_cumulative_capacity_reward=[]
        test_cumulative_fairness_reward=[]
        # print(test_noise)
        for i in range(test_noise.shape[0]):
            self.agent.replay_buffer.index=0
            capacity,fairness,actions=self.test_episode(test_noise[i])
            print(test_noise[i])
            if (test_noise[i][0]*100)%10==0:
                print(test_noise[i][0])
                test_cumulative_capacity_reward.append(capacity)
                test_cumulative_fairness_reward.append(fairness)
            else:
                cumulative_capacity_reward.append(capacity)
                cumulative_fairness_reward.append(fairness)
        # cumulative_capacity_reward, cumulative_fairness_reward = 
        cumulative_capacity_reward=np.array(cumulative_capacity_reward)
        cumulative_fairness_reward=np.array(cumulative_fairness_reward)
        np.save('./Exp/capacity_test',cumulative_capacity_reward)
        np.save('./Exp/fairness_test',cumulative_fairness_reward)
        self.agent.policy_net.eval=False
        plt.figure(figsize=(15,10))
        plt.scatter(cumulative_capacity_reward,cumulative_fairness_reward,color="DeepSkyBlue",label='Unseen weight')
        plt.scatter(test_cumulative_capacity_reward,test_cumulative_fairness_reward,color="OrangeRed",label='trained weight')
        # for i in range(cumulative_capacity_reward.shape[0]):
        #     plt.scatter(cumulative_capacity_reward[i],cumulative_fairness_reward[i],color=mcolors.TABLEAU_COLORS[colors[i%10]])
        #     plt.annotate("(%.2f,%.2f)"%(test_noise[i][0],test_noise[i][1]),xy=(cumulative_capacity_reward[i],cumulative_fairness_reward[i]))
        plt.tick_params(labelsize=16)
        plt.legend(prop={'size':16})
        plt.xlabel("Channel capacity (bps/Hz)",fontdict={'fontsize':24,})
        plt.ylabel("Fairness",fontdict={'fontsize':24,})
        # plt.xlabel("capacity")
        # plt.ylabel("fairness")
        plt.savefig('./Figure/test_MO_reward_4500.png')
        plt.close()


    def generate_episode(self,noise,sample=True):
        # generate a trajectory
        Channel,terminate,probs = [],[],[]
        instant_capacity_reward,instant_fairness_reward,Average_fairness_reward = [], [], []
        self.env.file_index=0
        self.env.Reset()
        terminated = False
        episode_capacity_reward = 0
        episode_fairness_reward = 0
        action_collector=[]
        while not terminated:
            episode_channel,  average_fairness_reward = self.env.get_state()
            actions, probability= self.agent.Pick_action(episode_channel, average_fairness_reward,noise,sample)
            capacity_reward, fairness_reward, terminated= self.env.Step(actions)
            # accumulate episode reward
            episode_capacity_reward += capacity_reward
            episode_fairness_reward += fairness_reward
            action_collector.append(actions)
            # collect data
            Channel.append(episode_channel)
            instant_capacity_reward.append(capacity_reward) 
            instant_fairness_reward.append(fairness_reward)
            Average_fairness_reward.append(average_fairness_reward)
            terminate.append(terminated)
            probs.append(probability)

        episode_batch = {}
        episode_batch['Channel'] = np.array(Channel)
        episode_batch['Average_fairness_reward'] = np.array(Average_fairness_reward)
        episode_batch['instant_capacity_reward'] = np.array(instant_capacity_reward)
        episode_batch['instant_fairness_reward'] = np.array(instant_fairness_reward)
        episode_batch['terminate'] = np.array(terminate)
        episode_batch['prob'] = probs
        episode_batch['noise'] = noise
        self.agent.Store_transition(episode_batch)

        return episode_capacity_reward, episode_fairness_reward, action_collector
    def test_episode(self,noise,sample=False):
        # generate a trajectory
        Channel,terminate,probs = [],[],[]
        instant_capacity_reward,instant_fairness_reward,Average_fairness_reward = [], [], []
        self.env.file_index=0
        self.env.Reset()
        terminated = False
        episode_capacity_reward = 0
        episode_fairness_reward = 0
        action_collector=[]
        while not terminated:
            episode_channel,  average_fairness_reward = self.env.get_state()
            actions, probability= self.agent.Pick_action(episode_channel, average_fairness_reward,noise,sample)
            capacity_reward, fairness_reward, terminated= self.env.Step(actions)
            # accumulate episode reward
            episode_capacity_reward += capacity_reward
            episode_fairness_reward += fairness_reward
            action_collector.append(actions)
            # collect data
            Channel.append(episode_channel)
            instant_capacity_reward.append(capacity_reward) 
            instant_fairness_reward.append(fairness_reward)
            Average_fairness_reward.append(average_fairness_reward)
            terminate.append(terminated)
            probs.append(probability)
        return episode_capacity_reward, episode_fairness_reward, action_collector


    def test_specify_model(self):
        self.agent.policy_net.load_state_dict(torch.load('./Model/3900_Agent_policy_net.pkl'))
        test_num=10
        color_set=['red','royalblue','darkorange','violet','springgreen']
        marker=['.','*','+','o']
        for i in range(test_num):
            self.env.read_training_data()
            self.env.Reset()
            terminated = False
            episode_capacity_reward = 0
            episode_fairness_reward = 0
            action_collector=[]
            while not terminated:
                episode_channel,  average_fairness_reward = self.env.get_state()
                actions, probability= self.agent.Pick_action(episode_channel, average_fairness_reward)
                capacity_reward, fairness_reward, terminated= self.env.Step(actions)
                episode_capacity_reward += capacity_reward
                episode_fairness_reward += fairness_reward
                action_collector.append(actions)
            print(i)
            print(episode_capacity_reward)
            plt.figure(figsize=(15,10))
            plt.ylim((-1,20))
            plt.yticks(np.arange(0,20,1))
            plt.grid()
            for x in range(len(action_collector)):
                for a in action_collector[x]:
                    plt.scatter(x,a,c=color_set[a%5],marker=marker[a//5])
            plt.title("schedule action")
            plt.savefig('./Figure/test_%d.png'%(i))
            plt.close()
    
    def save_model(self,epoch):
        # save model parameters
        policy_net_path = self.args.model_folder +  '/' +  '%d_Agent_policy_net_new.pkl'%epoch
        value_net_path = self.args.model_folder + '/' + '%d_Agent_value_net_new.pkl'%epoch
        value_net_2_path = self.args.model_folder + '/' + '%d_Agent_value_net_2_new.pkl'%epoch
        torch.save(self.agent.policy_net.state_dict(), policy_net_path)
        torch.save(self.agent.critic_net.state_dict(), value_net_path)
        torch.save(self.agent.critic_net_2.state_dict(), value_net_2_path)


    def load_model(self,epoch):
        policy_net_path = self.history +  '/' +  '%d_Agent_policy_net_new.pkl'%epoch
        value_net_path = self.history + '/' + '%d_Agent_value_net.pkl_new'%epoch
        # value_net_2_path = self.history + '/' + '%d_Agent_value_net_2.pkl'%epoch
        self.agent.policy_net.load_state_dict(torch.load(policy_net_path))
        self.agent.critic_net.load_state_dict(torch.load(value_net_path))
        # self.agent.critic_net_2.load_state_dict(torch.load(value_net_2_path))

    def plot_figure(self,epoch, Iteration_result, name):
        plt.figure()
        plt.plot(np.arange(len(Iteration_result))+1, Iteration_result)
        save_path = self.args.result_folder + '/' + name + '_result_%04d_new.png'%epoch
        np.save('./Exp/'+name+'_%d'%(epoch),Iteration_result)
        plt.savefig(save_path)
        plt.close()


    def test_model(self):
        self.agent.policy_net.load_state_dict(torch.load('./Model/3900_Agent_policy_net.pkl'))
        self.agent.policy_net.eval=True
        test_num=10
        for i in range(test_num):
            self.env.read_training_data()
            self.env.Reset()
            terminated = False
            episode_capacity_reward = 0
            episode_fairness_reward = 0
            action_collector=[]
            while not terminated:
                episode_channel,  average_fairness_reward = self.env.get_state()
                actions, probability= self.agent.Pick_action(episode_channel, average_fairness_reward)
                capacity_reward, fairness_reward, terminated= self.env.Step(actions)
                episode_capacity_reward += capacity_reward
                episode_fairness_reward += fairness_reward
                action_collector.append(actions)
            print(i)
            print(episode_capacity_reward)
            
test = Project()
test.Simulation()
#test.testUnseen()
# test.test_specify_model()
# test.test_model()
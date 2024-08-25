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

from model import Actor, Critic, Critic_2
from replay_buffer import ReplayBuffer


class Test:

    def __init__(self):
        self.args = get_A2C_args()
# create three folders which is used to store model, figure and tensorboard data

        # self.create_folder(self.args.result_folder)
        # self.create_folder(self.args.model_folder)
        # self.create_folder(self.args.vision_folder)

        # create a summary writer, which is used to record the loss value of individual agent

        self.args.writer = SummaryWriter(self.args.vision_folder)
        self.agent = Agent(self.args)
        self.agent.policy_net = self.load_policy_net()
        # define environment
        self.env = Environment(self.args)
        self.counter = 0 


    def load_policy_net(self):

        PATH =  './Model/Agent_policy_net.pkl'
        model = Actor(self.args)
        model.load_state_dict(torch.load(PATH))
        model.eval()

        return model


    def create_folder(self, folder_name):
        # create a folder, if folder exists, delete it first.
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)


    def Simulation(self,Round = 1):
        epoch_cumulative_reward =[]
        epoch_fairness_cumulative_reward =[]

        rewards_e = []
        rewards_f = []

        # optimal_reward = 0
        for ep in tqdm(range(Round)):
            episode_reward = []
            episode_fairness_reward = []
            self.env.read_training_data()
            for _ in range(self.args.episodes):
                # collect experience
                cumulative_reward, cumulative_fairness_reward = self.generate_episode()

                episode_reward.append(cumulative_reward)
                episode_fairness_reward.append(cumulative_fairness_reward)

            average_episode_reward = np.sum(episode_reward) / self.args.episodes
            average_episode_fairness_reward = np.sum(episode_fairness_reward) / self.args.episodes

            print("Epoch: {}, Current epoch average cumulative reward is: {};  fairness_reward is {}".format(ep, average_episode_reward,average_episode_fairness_reward))
            epoch_cumulative_reward.append(average_episode_reward)
            epoch_fairness_cumulative_reward.append(average_episode_fairness_reward)



        self.plot_figure(np.array(epoch_cumulative_reward), Round,"test_capacity")
        self.plot_figure(np.array(epoch_fairness_cumulative_reward), Round,"test_fairness")


        # self.save_model()


    def generate_episode(self):
        # generate a trajectory
        Channel, Average_reward, Average_fairness_reward = [],[],[]
        instant_reward, terminate, instant_fairness_reward = [], [], []

        probs = []

        self.env.Reset()
        terminated = False
        episode_reward = 0
        episode_fairness_reward = 0
        step = 1

        while not terminated:
            episode_channel, episode_average_reward, episode_average_fairness_reward = self.env.get_state()
            actions, prob= self.agent.Pick_action(episode_channel, episode_average_fairness_reward)
            print(actions)


            #np.random.randint(low=0,high size=10)

            # print(actions)
            # print(step)
            
            reward, terminated, fairness_reward = self.env.Step(actions,step)
   
            self.counter +=1

            episode_reward += reward
            episode_fairness_reward += fairness_reward

            step +=1 
 

        return episode_reward, episode_fairness_reward


    def plot_figure(self, Iteration_result, Round = 1,name = None):
        plt.figure()
        if not name:
            name = "default"

        plt.plot(np.arange(Round)+1,Iteration_result)
        save_path = self.args.result_folder + '/' + name + '.png'
        plt.savefig(save_path)
        plt.close()

test = Test()
test.Simulation(Round = 10)
        

#7750 2700


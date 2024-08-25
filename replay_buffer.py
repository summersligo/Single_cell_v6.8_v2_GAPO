# 这个函数用来存放episode的数据
import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.size = self.args.max_buffer_size
        self.episode_limit = 50
        self.sample_num=self.args.sample_num
        # memory management
        self.index = 0
        # 定义信道的维度
        self.obs_dim1 = self.args.channel_dim1
        self.obs_dim2 = self.args.channel_dim2
        # create the buffer to store info, 这个表示提前开辟好一个位置,然后将经验进行替换就可以了
        self.buffers = {'Channel': np.empty([self.size, self.episode_limit,  self.obs_dim1, self.obs_dim2]),
                        'Average_fairness_reward':np.empty([self.size, self.episode_limit,  self.obs_dim1]),
                        'instant_capacity_reward': np.empty([self.size, self.episode_limit]),
                        'instant_fairness_reward': np.empty([self.size, self.episode_limit]),
                        'terminate': np.empty([self.size, self.episode_limit]),
                        'noise': np.empty([self.size,2]),
                        'prob': []
                        }

        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        # batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            # store the informations
            self.buffers['Channel'][self.index] = episode_batch['Channel']
            self.buffers['Average_fairness_reward'][self.index] = episode_batch['Average_fairness_reward']
            self.buffers['instant_capacity_reward'][self.index] = episode_batch['instant_capacity_reward']
            self.buffers['instant_fairness_reward'][self.index] = episode_batch['instant_fairness_reward']
            self.buffers['terminate'][self.index] = episode_batch['terminate']
            self.buffers['noise'][self.index]=episode_batch['noise']
            self.buffers['prob'].append(episode_batch['prob'])
            # 这个地方的with lock操作表示在这个进程结束之前,其余的进程都是不可以运行的, 相当于lock.acquire() 以及结束之后运行lock.release()操作
            self.index += 1

    def sample(self):
        return self.buffers

    def reset_buffer(self):
        self.index = 0
        self.buffers = {'Channel': np.empty([self.size, self.episode_limit,  self.obs_dim1, self.obs_dim2]),
                        'Average_fairness_reward':np.empty([self.size, self.episode_limit,  self.obs_dim1]),
                        'instant_capacity_reward': np.empty([self.size, self.episode_limit]),
                        'instant_fairness_reward': np.empty([self.size, self.episode_limit]),
                        'terminate': np.empty([self.size, self.episode_limit]),
                        'noise': np.empty([self.size,2]),
                        'prob': []
                        }

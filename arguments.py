import argparse
import torch

def get_common_args():
    flag = torch.cuda.is_available()
    #flag = False
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--user_numbers', type=int, default=10, help='the number of users of single cell')
    parser.add_argument('--user_antennas', type=int, default=2, help='the number of user antennas')
    parser.add_argument('--bs_antennas',type=int, default=8, help='the number of base station antennas')
    parser.add_argument('--cuda', type=bool, default=flag, help='whether to use the GPU')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount ratio')
    parser.add_argument('--noise_power', type=float, default=5.6921e-15 / 50, help='the noise power')
    parser.add_argument('--transmit_power', type=float, default=0.25, help='the total transmit power of base station')
    parser.add_argument('--tau', type=float, default=0.8, help='the filter coefficient')
    parser.add_argument('--t_c', type=float, default=0.9, help='the update coefficient of average fairness reward')
    parser.add_argument('--TTI_length',type=int,default=100, help='the TTI length')
    parser.add_argument('--data_folder', type=str, default="./data", help='the original data folder')
    # 定义图片存储的文件夹
    parser.add_argument('--result_folder', type=str, default="./Figure", help='the result folder of figure and data')
    # 定义模型存储的文件夹
    parser.add_argument('--model_folder', type=str, default='./Model', help='the folder of pretrained model')
    # 定义tensorboard的打开文件夹
    parser.add_argument('--vision_folder', type=str, default='./Exp', help='the folder of tensorboard result')
    args = parser.parse_args()
    return args

def get_agent_args():
    args = get_common_args()
    # 计算能够支持最大流的数目
    total_user_antennas = args.user_antennas * args.user_numbers
    max_stream = min(total_user_antennas, args.bs_antennas)
    args.max_stream = max_stream
    args.channel_dim1 = args.user_numbers * args.user_antennas
    args.channel_dim2 = args.bs_antennas * 2
    return args

def get_A2C_args():
    args = get_agent_args()
    # 定义epoch的数目
    args.epoches = 40000
    # 定义episode的数目
    args.episodes = 5
    args.noise_dim=4
    w1=torch.arange(0,1.05,0.05)
    args.noise=torch.stack((w1,1-w1),1)
    args.sample_num=args.noise.shape[0]
    args.epsilon = 0.3
    args.min_epsilon = 0
    args.epsilon_decay = (args.epsilon-args.min_epsilon) / (0.9*args.epoches)
    actor_lr = 1e-3
    actor_lr_decay = 1e-4
    args.actor_lr = actor_lr
    args.actor_lr_decay = actor_lr_decay
    # 定义策略网络的神经元个数
    args.rnn_hidden = 64
    # 定义pointer network的权重向量的长度
    args.weight_dim = 32
    
    args.embedding_dim = 2
    # 由于是按照天线进行调度的,所以每一根天线实部和虚数部分分开,得到的特征长度为32
    args.flatten_dim = 2* args.bs_antennas
    # 定义数据嵌入的维度
    args.embedding_dim = 2
    # 进行一次升维度
    args.high_space_dim = 32
    # flatten之后的信道矩阵进行低维的嵌入操作之后, 以及与average user sum rate进行拼接,得到状态维度是user_num * 3, 然后升维
    args.rnn_input_dim = 64

    # 定义critic网络的相关参数
    # 定义几个仿射层
    args.hidden_dim = [32,64,128,32]
    args.fc_layer_number = 4
    critic_lr = 1e-3
    critic_lr_decay = 1e-4
    args.critic_lr = critic_lr
    args.critic_lr_decay = critic_lr_decay
    args.update_times = 5
    # 定义GAE参数
    args.GAE_factor = 1e-3
    # 定义buffer的size
    args.max_buffer_size = args.episodes
    return args





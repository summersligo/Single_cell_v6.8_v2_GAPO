# 这个文件用来对信道矩阵进行数据的预处理
import os
import multiprocessing
from multiprocessing import Pool
import shutil
import numpy as np
from scipy.io import loadmat

def create_data_folder(data_folder):
    # 这个函数用来判断文件夹存不存在，如果存在，则删除，然后创建，如果不存在，则直接创建
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder)

def preprocess_data(active_file, user_antenna_number):
    # 这个文件就是对信道文件进行处理，信道文件是.mat文件
    # 首先读取这个文件
    H_file = loadmat(active_file)
    H_matrix = H_file['H_DL_File']
    # 9维的信道矩阵参数说明
    # 第一个维度是RB的数量，这里通过平均处理
    # 第二个是目的基站的数量，第三个是目的扇区的数量
    # 第四个维度是总共天线的数目，第五个维度表示的每个用户的接收天线数目
    # 第六个维度表示的是源基站 第七个维度表示的是源扇区
    # 第八个维度表示的基站的发射天线，第九个维度表示的是TTI
    tilde_H = np.mean(H_matrix, 0).squeeze()
    # 现在这个tilde_H变量的形状为(3, 20, 3, 16, 50)
    # 这个地方集成了三个episode的数据
    cell_number = tilde_H.shape[0]
    antenna_number = tilde_H.shape[1]
    base_station_number = tilde_H.shape[3]
    TTI_number = tilde_H.shape[4]
    real_user_number = int(antenna_number/user_antenna_number)
    index = [i + real_user_number * j for i in range(real_user_number) for j in range(user_antenna_number)]
    # 这个地方生成一个array，将信道矩阵进行对齐
    result = []
    for cell_index in range(cell_number):
        temp_result = tilde_H[cell_index,np.array(index),cell_index,:,:].squeeze()
        # 这个temp_result的维度是20*16*50
        # temp_result_reshape = temp_result.reshape(real_user_number, user_antenna_number,  base_station_number, TTI_number)
        result.append(np.concatenate((np.real(temp_result), np.imag(temp_result)), axis=1)) 
        # 得到一个长度为三的列表，列表中的每一个元素都是长度为20*32*50的矩阵
    return result

def preprocess_single_file(folder_name, data_folder, user_antenna_number=2):
    # 由于每一个文件夹里面有4个CH开头的mat文件，直接遍历这三个文件
    file_list = sorted(os.listdir(folder_name))
    file_result = []
    for file_name in file_list:
        if "CH3D" in file_name:
            target_file_position = folder_name + '/' + file_name
            TTI_file_result =  preprocess_data(target_file_position, user_antenna_number)
            file_result.append(TTI_file_result)
    # 得到的是一个四行三列的二维列表
    # 重新进行拼接，得到三个信道文件
    file_number = len(file_result)
    cell_number = len(file_result[0])
    file_result = np.array(file_result) # (4,3,20,32,50)
    # 上面两个变量，一个表示的是CH3D信道文件的数量，另外一个表示的是小区的数目
    for cell_index in range(cell_number):
        temp_dataset = tuple([file_result[CH_index , cell_index, :, :, :] for CH_index in range(file_number)])
        episode_data = np.concatenate(temp_dataset, axis=2)
        save_name = data_folder + '/' + folder_name.split('/')[-1] + '_' + str(cell_index)
        if os.path.exists(save_name + '.npy'):
            os.remove(save_name + '.npy')
        np.save(save_name+ '.npy', episode_data)


if __name__ =='__main__':
    data_folder = './data'
    source_data_folder = '../../backup/'
    folder_name = os.listdir(source_data_folder)
    # 创建一个文件夹用来存放处理好之后的数据
    create_data_folder(data_folder)
    relative_position = [source_data_folder + active_folder_name for active_folder_name in folder_name]
    # 由于数据样本中有一部分是10个用户的，一部分是15个用户的
    # 因此一个进程处理两个数据文件
    workers = os.cpu_count() - 1
    # workers = 1
    pool = Pool(processes=workers)
    folder_number = len(relative_position)
    for folder_index in range(folder_number):
        pool.apply_async(preprocess_single_file, (relative_position[folder_index], data_folder))
        print(relative_position[folder_index])
    pool.close()
    pool.join()
    # preprocess_single_file(relative_position[0])

import pandas as pd
import numpy as np

#只运行一次
def save_normalized_data(path):
    #load data from path
    data = np.loadtxt(path)
    data = normalize(data)
    np.savetxt('normalized.txt', data)
#只运行一次
def normalize(data):
    #生成数据中的最小值和最大值的索引序列
    min = np.argmin(data, axis=0)
    max = np.argmax(data, axis=0)
    min_value = []
    max_value = []
    for i in range(len(min)):
        min_value.append(data[min[i]][i])
        max_value.append(data[max[i]][i])
    min_value = np.array(min_value)
    max_value = np.array(max_value)
    for i in range(len(data)):
        data[i] = (data[i] - min_value) / (max_value - min_value)
    return data

#为train.py所调用
#加载train或者test数据，返回可用的X, y型
def load_data(path):
    data = np.loadtxt(path).tolist()
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i][0:10])
        y.append(data[i][10:])
    return np.array(X), np.array(y)

#读取数据，分离train和test
def split(test_sample_percentage,path):
    data = np.loadtxt(path)
    shuffled_data = np.random.permutation(data)
    test_sample_index = -1 * int(test_sample_percentage * float(len(shuffled_data)))
    train, test = shuffled_data[:test_sample_index], shuffled_data[test_sample_index:]
    np.savetxt('../data/UCI/train_airoil_new.txt', train)
    np.savetxt('../data/UCI/test_airoil_new.txt', test)


#generator batch data with shuffled
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batchs_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        #shuffled data every epoch
        if shuffle:
            data_shufled = np.random.permutation(data)
        else:
            data_shufled = data
        for num_batch in range(num_batchs_per_epoch):
            start_index = num_batch * batch_size
            end_index = min((num_batch + 1) * batch_size, data_size)
            yield data_shufled[start_index : end_index]

#split(0.2, 'normalized.txt') 34/137
#_, y = load_data('normalized.txt')

# 交叉特征
def save_featurecrosses_data(path):
    data = np.loadtxt(path)
    data_new = []
    #进行两两交叉3
    for i in range(len(data)):
        temp = []
        for j in range(len(data[i])-2):
            for k in range(j+1,len(data[i])-1):
                temp.append(data[i][j]*data[i][k])
        temp.append(data[i][-1])
        data_new.append(temp)
    data_new = normalize(data_new)
    data_new = np.array(data_new)

    np.savetxt('../data/UCI/airoil_new.txt', data_new)

#获取输出列的最大值和最小值，求逆归一化
def get_min_max(path):
    data = np.loadtxt(path)
    y = data[:,5]
    min_index = np.argmin(y)
    max_index = np.argmax(y)
    min_value = y[min_index]
    max_value = y[max_index]
    return min_value, max_value


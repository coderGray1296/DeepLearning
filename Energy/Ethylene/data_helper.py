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
        X.append(data[i][0:5])
        y.append(data[i][5:])
    return X, y

#读取数据，分离train和test
def split(test_sample_percentage,path):
    data = np.loadtxt(path)
    shuffled_data = np.random.permutation(data)
    test_sample_index = -1 * int(test_sample_percentage * float(len(shuffled_data)))
    train, test = shuffled_data[:test_sample_index], shuffled_data[test_sample_index:]
    np.savetxt('train.txt', train)
    np.savetxt('test.txt', test)


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

#split(0.2, 'normalized.txt') /137
#_, y = load_data('normalized.txt')


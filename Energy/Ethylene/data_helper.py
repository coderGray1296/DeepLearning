import pandas as pd
import numpy as np


def save_normalized_data(path):
    #load data from path
    data = np.loadtxt(path)
    data = normalize(data)
    np.savetxt('normalized.txt', data)

def normalize(data):
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

def load_data(path):
    data = np.loadtxt(path).tolist()
    X = []
    y = []
    for i in range(len(data)):
        X.append(data[i][0:5])
        y.append(data[i][5:])
    return X, y

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    


#_, y = load_data('normalized.txt')

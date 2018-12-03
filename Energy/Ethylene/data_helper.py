import pandas as pd
import numpy as np


def load_data(path):
    #load data from path
    data = np.loadtxt(path)
    data = normalize(data)
    np.savetxt('normalized.txt', data,)

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

load_data('data.txt')
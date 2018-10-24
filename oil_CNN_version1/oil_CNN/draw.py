import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class draw():
    def __init__(self):
        self.path = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/draw/'
        self.data = []
        self.time = []

    def get_data(self):
        file = open(self.path+'qiang.txt')
        DF = pd.read_csv(file, sep='\t', header=None)
        self.data = list(DF[1])
        self.time = list(DF[0])

    def draw_picture(self,i):
        #t = np.arange(len(self.time))
        t = np.arange(1024)
        plt.figure(figsize=(10, 6))
        plt.plot(t, np.array(self.data), 'r-')
        plt.xlabel("time")
        plt.ylabel("key")
        plt.legend(['data', 'validation'], loc='upper right')
        plt.savefig('/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/static/image/'+str(i)+'.png')
        #plt.show()
'''
d = draw()
d.get_data()
d.draw_picture()
'''
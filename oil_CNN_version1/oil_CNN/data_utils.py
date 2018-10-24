import numpy as np
import os
import pandas as pd

class util():
    def __init__(self):
        self.channels = 1
        self.path = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN'
        self.train_X = 'train_X.txt'
        self.test_X = 'test_X.txt'
        self.train_y = 'train_y.txt'
        self.test_y = 'test_y.txt'
        self.ts_len = 1024

    def read_data(self,sub_dir = 'train'):
        if sub_dir == 'train':
            train_path = os.path.join(self.path, sub_dir)
            train_X_path = os.path.join(train_path, self.train_X)
            train_y_path = os.path.join(train_path,self.train_y)
            DF_y = pd.read_csv(train_y_path, header=None)
            X = np.zeros([len(DF_y), self.ts_len, self.channels])
            DF_X = pd.read_csv(train_X_path, delim_whitespace=True, header=None)
            X[:, :, self.channels - 1] = DF_X.as_matrix()
            y = DF_y[0].values
        elif sub_dir == 'test':
            test_path = os.path.join(self.path, sub_dir)
            test_X_path = os.path.join(test_path, self.test_X)
            test_y_path = os.path.join(test_path, self.test_y)
            DF_y = pd.read_csv(test_y_path, header=None)
            X = np.zeros([len(DF_y), self.ts_len, self.channels])
            DF_X = pd.read_csv(test_X_path, delim_whitespace=True, header=None)
            X[:, :, self.channels - 1] = DF_X.as_matrix()
            y = DF_y[0].values
        return X,y

    def get_batches(self, X, y, batch_size = 10):
        n_batches = len(X) // batch_size
        X = X[:n_batches * batch_size]
        y = y[:n_batches * batch_size]
        for b in range(0, len(X), batch_size):
            yield X[b:b+batch_size], y[b:b+batch_size]

    #onehot编码
    def one_hot(self, labels, classes = 3):
        expansion = np.eye(classes)
        print(expansion.shape)
        y = expansion[:,labels - 1].T
        print(y.shape)
        return y

'''
sub_dir = 'train'
U = util()
X,y = U.read_data(sub_dir)
print(len(X))
print(X.shape)
print(y.shape)
'''

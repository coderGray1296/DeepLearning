import numpy as np


class test():
    def one_hot(self, labels, classes = 3):
        expansion = np.eye(classes)
        print(expansion.shape)
        y = expansion[:, labels - 1].T
        print(y.shape)
        return y
    def haha(self):
        x = 1e-5
        print(x*1000)



t = test()
t.haha()
'''
train_y = []
train_y.append(3)
train_y.append(1)
train_y.append(2)
train_y.append(3)
train_y = np.array(train_y).T
train_y = t.one_hot(train_y,classes=3)
print(train_y)
'''






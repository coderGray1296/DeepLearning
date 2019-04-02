import numpy as np
import matplotlib.pyplot as plt
import data_helper

class HiddenLayer:
    def __init__(self, x, num):
        #样本个数为row，样本特征数为columns
        row = x.shape[0]
        columns = x.shape[1]

        rnd = np.random.RandomState(4444)
        self.w = rnd.uniform(-1,1,(columns,num))
        self.b = np.zeros([row,num],dtype=float)

        for i in range(num):
            rand_b = rnd.uniform(-0.4,0.4)
            for j in range(row):
                self.b[j,i] = rand_b

        h = self.sigmoid(np.dot(x,self.w)+self.b)
        #利用numpy求h的广义逆矩阵
        self.H = np.linalg.pinv(h)

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    #T为标签np.array
    def regressor_train(self, T):
        T = T.reshape(-1,1)
        self.beta = np.dot(self.H,T)
        return self.beta

    def regressor_test(self, test_data):
        b_row = test_data.shape[0]
        h = self.sigmoid(np.dot(test_data, self.w) + self.b[:b_row,:])
        result = np.dot(h,self.beta)
        return result
    def relative_error(self, test_y, output_y):
        self.losses = abs(output_y / test_y - 1)
        print(self.losses.shape)
        return np.mean(self.losses)

train_x, train_y = data_helper.load_data('train_new.txt')
test_x, test_y = data_helper.load_data('test_new.txt')
elm = HiddenLayer(train_x, 10)
elm.regressor_train(train_y)
output_y = elm.regressor_test(test_x)
loss = elm.relative_error(test_y, output_y)
print(loss)

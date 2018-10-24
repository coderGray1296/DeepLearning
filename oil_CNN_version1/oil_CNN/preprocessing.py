import numpy as np
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from sklearn.model_selection import  train_test_split
import tensorflow as tf
from data_utils import util



class preprocessing():

    def __init__(self):
        self.src_path = '/Users/codergray/Desktop/study/graduate-project/J16-source_data'
        self.dest_path_train = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/train'#'/Users/codergray/Desktop/study/graduate-project/train'
        self.dest_path_test = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/test'#'/Users/codergray/Desktop/study/graduate-project/test'
        self.train_file_X_name = 'train_X.txt'
        self.train_file_y_name = 'train_y.txt'
        self.test_file_X_name = 'test_X.txt'
        self.test_file_y_name = 'test_y.txt'
        self.X_name = 'X'
        self.y_name = 'labels'
        self.threshold = 50
        self.features = 1024
        self.min  = 2
        self.max = 30

        #autoencoder
        self.learning_rate = 0.01
        self.batch_size = 35
        self.display_step = 1
        self.training_epochs = 20
        self.len_input = 33599
        self.n_hidden_1 = 4096
        self.n_hidden_2 = 1024
        self.weights = {
            'encoder_h1': tf.Variable(tf.random_normal([self.len_input, self.n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.len_input])),
        }
        self.biases = {
            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([self.len_input])),
        }

    def get_labels(self):
        paths_labels = {}
        labels = {}
        for dir_path, dir_names, file_names in os.walk(self.src_path):
            labels = self.classify(files=file_names, threshold=self.threshold)
        labels = self.to_digit(labels)
        #返回的数据key是绝对路径+文件名，方便下一步读文件
        for key, value in labels.items():
            paths_labels[os.path.join(self.src_path, key)] = value
        return paths_labels


    #去掉标签记录数小于阈值的样本数据和标签，返回一个dict
    def classify(self,files,threshold):
        labels = {}
        for file in files:
            labels[file] = str(file).split('-')[0]
        list_values = list(labels.values())
        labels_set = set(list_values)
        cp_labels = labels.copy()
        for item in labels_set:
            if list_values.count(item)<threshold:
                for key,value in labels.items():
                    if labels[key] == str(item):
                        del cp_labels[key]
        return cp_labels

    #将字符串标签转换成数字形式
    def to_digit(self,labels):
        digit_labels = {}
        classes = set(labels.values())
        class_digits = {}
        i = 1
        #编号从1开始
        for item in sorted(list(classes)):
            class_digits[item] = i
            i += 1
            print(str(item)+str(i-1))
        for key,value in labels.items():
            digit_labels[key] = class_digits[labels[key]]
        return digit_labels

    #读文件
    def read_file(self,paths_labels):
        X = []
        y = []
        for k, v in paths_labels.items():
            file = open(k)
            DF = pd.read_csv(file, sep='\t', header=None)
            DF = DF.loc[DF.loc[:, 0] > self.min, :]
            DF = DF.loc[DF.loc[:, 0] < self.max, :]
            L = list(DF[1])
            if len(L) >= self.features:
                #print("L:"+str(len(L)))
                X.append(L)
                y.append(v)
        return X, y

    def get_min_and_max(self,X):
        min = 100000
        max = 0
        print(len(X))
        for i in range(len(X)):
            print("i:"+str(i)+"len:"+str(len(X[i])))
            if(len(X[i]) < min):
                min = len(X[i])
            if(len(X[i]) > max):
                max = len(X[i])
        print("min:"+str(min))
        print("max:"+str(max))

    #特征选择，卡方检验
    def select_features(self,X,y):
        selector = SelectKBest(score_func=f_classif, k=self.features)
        X = selector.fit_transform(X, y)
        return X, y


    def select_max_featres(self,X,y):
        X_ = []
        for i in range(len(X)):
            Xtemp = []
            #将list分组，求每组的元素个数
            interval = len(X[i])//self.features
            for j in range(self.features):
                start = j * interval
                end = start + interval
                temp = max(X[i][start:end])
                Xtemp.append(temp)
            print("Xtemp"+str(len(Xtemp)))
            X_.append(Xtemp)
        print(len(X_))
        return X_, y


    #划分训练集和测试集
    def split_train_test(self, X, y, test_size = 0.2, sd = 1):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = sd)
        return X_train, X_test, y_train, y_test

    #针对训练集进行数据标准化
    def standardize(self, data):
        std_data = (data - np.mean(data))/np.std(data)
        return std_data

    def write_data_to_file(self, X, y):
        X_train, X_test, y_train, y_test = pp.split_train_test(X, y, test_size=0.1, sd=1)
        train_file_X = open(os.path.join(self.dest_path_train, self.train_file_X_name), 'w')
        train_file_y = open(os.path.join(self.dest_path_train, self.train_file_y_name), 'w')
        for i in range(len(X_train)):
            train_std_X = self.standardize(X_train[i])
            for j in range(len(train_std_X)):
                train_file_X.write('{:9f}'.format(train_std_X[j]) + '\t')
            train_file_X.write('\n')
            train_file_y.write(str(y_train[i]) + '\n')

        test_file_X = open(os.path.join(self.dest_path_test, self.test_file_X_name), 'w')
        test_file_y = open(os.path.join(self.dest_path_test, self.test_file_y_name), 'w')
        for i in range(len(X_test)):
            test_std_X = self.standardize(X_test[i])
            for j in range(len(test_std_X)):
                test_file_X.write('{:9f}'.format(test_std_X[j]) + '\t')
            test_file_X.write('\n')
            test_file_y.write(str(y_test[i]) + '\n')


    #autoencoder:33599-->4096-->1024-->4096-->33599
    def placeholder(self):
        self.inputs = tf.placeholder(tf.float32, [None,self.len_input])

    #构建编码器
    def encoder(self,x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['encoder_h1']),
                                       self.biases['encoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),
                                       self.biases['encoder_b2']))
        return layer_2

    #构建解码器
    def decoder(self,x):
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self.weights['decoder_h1']),
                                       self.biases['decoder_b1']))
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),
                                       self.biases['decoder_b2']))
        return layer_2


    def autoencoder(self,x_,y_):
        self.placeholder()

        U = util()
        # 构建模型
        encoder_op = self.encoder(self.inputs)
        decoder_op = self.decoder(encoder_op)
        #print(x_.shape)
        # 预测
        y_pred = decoder_op
        y_true = self.inputs

        # 定义代价函数和优化器
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))  # 最小二乘法
        optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)

        with tf.Session() as sess:
            #初始化
            init = tf.global_variables_initializer()
            sess.run(init)
            # 首先计算总批数，保证每次循环训练集中的每个样本都参与训练，不同于批量训练
            for epoch in range(self.training_epochs):
                for x, y in U.get_batches(x_, y_, batch_size=self.batch_size):
                    # print("epoch:"+str(epoch)+"iter:
                    _, c, e_op= sess.run([optimizer, cost, encoder_op], feed_dict={self.inputs: x})
                    print(e_op.shape)
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))
            #restore the result 445*1024
            result = sess.run(encoder_op, feed_dict={self.inputs: x_})
            print("Optimization Finished!")
        return result, y_

if __name__ == '__main__':
    print('preprocessing data')
    pp = preprocessing()
    paths_labels = pp.get_labels()
    print(len(paths_labels))
    for k,v in paths_labels.items():
        print(str(k)+':'+str(v))
    X_, y_ = pp.read_file(paths_labels)
    xx, yy = pp.autoencoder(X_,y_)
    #pp.get_min_and_max(X_)
    #X, y = pp.select_max_featres(X_,y_)
    #X, y = pp.select_features(X_, y_)
    pp.write_data_to_file(xx, yy)
    print('preprocessing data finished')
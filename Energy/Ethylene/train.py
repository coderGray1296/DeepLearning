import tensorflow as tf
import numpy as np
import time
import datetime
import os
from model import CNN
import data_helper
from tensorflow.contrib import learn

#设置训练集和测试集为4：1
test_sample_percentage = 0.2
data_path = 'normalized.txt'

filter_sizes = [3, 4, 5]
num_filters = 10
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 16
num_epochs = 10
#Evaluate model on dev set after this many steps (default: 100)
evaluate_every = 100
#Save model after this many steps (default: 100)
checkpoint_every = 100
#Number of checkpoints to store (default: 5)
num_checkpoints = 5

#Data Preparation
print('Loading data...')
X, y = data_helper.load_data(data_path)

# Split train/test set
test_sample_index =  -1 * int(test_sample_percentage * float(len(y)))
x_train, x_test = X[:test_sample_index], X[test_sample_index:]
y_train, y_test = y[:test_sample_index], y[test_sample_index:]

#Training
#==========================>
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = CNN(
            input_size = x_train.shape[1],
            output_size = y_train.shape[1],
            filter_sizes = filter_sizes,
            num_filters = num_filters,
            l2_reg_lambda = l2_reg_lambda
        )
        #定义training步骤
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # print(global_step)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            


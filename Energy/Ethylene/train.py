import tensorflow as tf
import numpy as np
import time
import datetime
import os
from model import CNN
import data_helper
from tensorflow.contrib import learn
import draw

#设置训练集和测试集为4：1
test_sample_percentage = 0.2
data_path = 'normalized.txt'

filter_sizes = [3, 2]
num_filters = 10
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 16
num_epochs = 100
#Evaluate model on dev set after this many steps (default: 100)
evaluate_every = 20
#Save model after this many steps (default: 100)
checkpoint_every = 100
#Number of checkpoints to store (default: 5)
num_checkpoints = 5

#Data Preparation
print('Loading data...')
x_train, y_train = data_helper.load_data('train.txt')
x_test, y_test = data_helper.load_data('test.txt')

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
        print('cnn.shape:')
        print(cnn.drop.shape)
        #定义training步骤
        global_step = tf.Variable(0, name="global_step", trainable=False)
        # print(global_step)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # 初始化所有变量
        sess.run(tf.global_variables_initializer())

        #training step
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return loss, accuracy

        #testing step
        def test_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            print(x_batch.shape)
            step, losses, loss, accuracy = sess.run([global_step, cnn.losses, cnn.loss, cnn.accuracy],
                                                       feed_dict)
            #time_str = datetime.datetime.now().isoformat()
            #print("{}: step {},loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return losses

        batches = data_helper.batch_iter(list(zip(x_train, y_train)), batch_size, num_epochs)

        train_loss_all = []
        train_accuracy_all = []
        test_loss_all = []
        test_accuracy_all = []
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            loss_train, accuracy_train = train_step(x_batch, y_batch)
            train_loss_all.append(loss_train)
            train_accuracy_all.append(accuracy_train)
            current_step = tf.train.global_step(sess, global_step)
            '''
            #每evaluate_every进行一次测试
            if current_step % evaluate_every == 0:
                print('\nTesting:')
                loss_test, accuracy_test = test_step(x_test, y_test)
                test_loss_all.append(loss_test)
                test_accuracy_all.append(accuracy_test)
                print("")
            '''


        losses = test_step(x_test, y_test)
        print(losses)

        # draw picture for loss and accuracy of test and train
        draw.draw_picture('train', train_accuracy_all, train_loss_all)
        #draw.draw_picture('test', test_accuracy_all, test_loss_all)

        print('modelling finished!')

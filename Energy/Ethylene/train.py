import tensorflow as tf
import numpy as np
import time
import datetime
import os
from model import CNN
from model_new import CNN_NEW
import data_helper
from tensorflow.contrib import learn
import draw

#设置训练集和测试集为4：1
test_sample_percentage = 0.2

filter_sizes = [5, 2]
num_filters = [3, 5]
dropout_keep_prob = 0.5
l2_reg_lambda = 0.0

batch_size = 16
num_epochs = 200
#Evaluate model on dev set after this many steps (default: 100)
evaluate_every = 10
#Save model after this many steps (default: 100)
checkpoint_every = 100
#Number of checkpoints to store (default: 5)
num_checkpoints = 5

#Data Preparation
print('Loading data...')
x_train, y_train = data_helper.load_data('train_new.txt')
x_test, y_test = data_helper.load_data('test_new.txt')


#Training
#==========================>
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        cnn = CNN_NEW(
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
        
        #training step
        def train_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: dropout_keep_prob
            }
            _, step, losses, loss, accuracy = sess.run(
                [train_op, global_step, cnn.losses, cnn.loss, cnn.accuracy], feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return loss, accuracy
        

        def test_step(x_batch, y_batch):
            feed_dict = {
                cnn.input_x: x_batch,
                cnn.input_y: y_batch,
                cnn.dropout_keep_prob: 0.5
            }
            losses, loss, accuracy = sess.run([cnn.losses, cnn.loss, cnn.accuracy],
                                              feed_dict)
            # time_str = datetime.datetime.now().isoformat()
            # print("{}: step {},loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return loss, accuracy


        model_saver = tf.train.Saver()

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
            if current_step % evaluate_every == 0:
                loss_test, accuracy_test = test_step(x_test, y_test)
                test_loss_all.append(loss_test)
                test_accuracy_all.append(accuracy_test)
        #losses = test_step(x_test, y_test)
        #print(losses)
        model_saver.save(sess, '../checkpoint/result.ckpt', global_step=global_step)

        # draw picture for loss and accuracy of test and train
        draw.draw_picture('train', train_accuracy_all, train_loss_all)
        draw.draw_picture('test', test_accuracy_all, test_loss_all)
        '''
        ckpt = tf.train.get_checkpoint_state("../checkpoint/")
        model_saver.restore(sess, ckpt.model_checkpoint_path)
        #losses = test_step(x_test, y_test)
        #print(losses)

        for i in range(10):
            result = []
            loss, _ = test_step(x_test, y_test)
            print(loss)
        '''

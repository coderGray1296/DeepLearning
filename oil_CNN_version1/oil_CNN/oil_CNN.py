import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from data_utils import util
import matplotlib.pyplot as plt

class oil_cnn():
    def __init__(self):
        self.path = '/Users/codergray/PycharmProjects/oil_CNN_version1'
        self.threshod = 50
        self.seq_len = 1024
        self.batch_size = 50
        self.keep_prob = 0.5
        self.learning_rate = 1e-5
        self.epochs = 4000 #4000
        self.channels = 1
        self.classes = 3
        self.train_loss = []
        self.train_acc = []
        self.validation_loss = []
        self.validation_acc = []
        self.iter = 1
        self.prob = 0
        self.logits = 0
        self.log_path = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/log'
        self.checkpoints_path = '/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/checkpoints'


    def util_placeholder(self, graph):
        with graph.as_default():
            self.inputs = tf.placeholder(tf.float32,[None,self.seq_len,self.channels],name='input')
            self.labels = tf.placeholder(tf.float32,[None,self.classes],name='labels')
            self.keep_prob_ = tf.placeholder(tf.float32,name='prob')
            self.learning_rate_ = tf.placeholder(tf.float32,name='learning_rate')

    def create_network(self, graph):

        #(batch,1024,1) --> (batch,256,4)
        conv1 = tf.layers.conv1d(inputs=self.inputs,filters=4,kernel_size=2,strides=1,padding='SAME',activation=tf.nn.relu)
        print(conv1.shape)
        max_pooling_1 = tf.layers.max_pooling1d(inputs=conv1,pool_size=4,strides=4,padding='SAME')
        print(max_pooling_1.shape)

        # (batch,256,4) --> (batch,64,16)
        conv2 = tf.layers.conv1d(inputs=max_pooling_1,filters=16,kernel_size=2,strides=1,padding='SAME',activation=tf.nn.relu)
        print(conv2.shape)
        max_pooling_2 = tf.layers.max_pooling1d(inputs=conv2,pool_size=4,strides=4,padding='SAME')
        print(max_pooling_2.shape)

        # (batch,64,16) --> (batch,16,64)
        conv3 = tf.layers.conv1d(inputs=max_pooling_2,filters=64,kernel_size=2,strides=1,padding='SAME',activation=tf.nn.relu)
        print(conv3.shape)
        max_pooling_3 = tf.layers.max_pooling1d(inputs=conv3,pool_size=4,strides=4,padding='SAME')
        print(max_pooling_3.shape)

        # (batch,64,64) --> (batch,16,256)
        #conv4 = tf.layers.conv1d(inputs=max_pooling_3,filters=256,kernel_size=2,strides=1,padding='SAME',activation=tf.nn.relu)
        #print(conv4.shape)
        #max_pooling_4 = tf.layers.max_pooling1d(inputs=conv4,pool_size=4,strides=4,padding='SAME')
        #print(max_pooling_4.shape)

        # Flatten and add dropout
        flat = tf.reshape(max_pooling_3,(-1,16*64))
        flat = tf.nn.dropout(flat, keep_prob=0.7)

        # Predictions`
        self.logits = tf.layers.dense(flat, self.classes)
        #print(logits)

        # Cost function and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels))
        self.prob = tf.nn.softmax(self.logits)
        #self.prob=tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.labels)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        # Accuracy
        self.correct_pred = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.labels,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred,tf.float32),name='accuracy')

    def exec(self,graph,x_,y_):
    #def exec(self,graph):
        '''
        print('data splitting')
        sub_dir = 'train'
        U = util()
        X, y = U.read_data(sub_dir)

        #在训练集中生成验证集，按照1：3
        train_X, vld_X ,train_y, vld_y = train_test_split(X, y, test_size=0.25, random_state=1)
        train_y = U.one_hot(train_y, classes=self.classes)
        vld_y = U.one_hot(vld_y, classes=self.classes)
        print('splitting data finished')
        '''
        with tf.Session(graph=graph) as sess:
            oil_cnn.util_placeholder(self, graph)
            oil_cnn.create_network(self, graph)
            print('initializing network finished')

            # 生成checkpoints，Saver
            with graph.as_default():
                self.saver = tf.train.Saver()
            '''
            writer = tf.summary.FileWriter(self.log_path, sess.graph)
            tf.summary.scalar('loss', self.cost)
            tf.summary.scalar('accuracy', self.accuracy)
            merged = tf.summary.merge_all()

            #初始化
            init=tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.epochs):
                for x,y in U.get_batches(train_X,train_y,batch_size=self.batch_size):
                    feed={self.inputs:x,self.labels:y,self.keep_prob_:self.keep_prob,self.learning_rate_:self.learning_rate}
                    loss,_,acc,prob = sess.run([self.cost, self.optimizer, self.accuracy, self.prob], feed_dict=feed)
                    #print(prob)
                    self.train_acc.append(acc)
                    self.train_loss.append(loss)
                    #print("epoch:"+str(epoch)+"iter:"+str(iter))
                    if self.iter%5 == 0:
                        print("Epoch:{}/{}".format(epoch, self.epochs),
                              "Iteration:{:d}".format(self.iter),
                              "Train loss:{:6f}".format(loss),
                              "Train acc:{:.6f}".format(acc))

                    if self.iter%10 == 0:
                        vld_acc = []
                        vld_loss = []
                        for x_v, y_v in U.get_batches(vld_X, vld_y,batch_size=self.batch_size):
                            feed = {self.inputs:x_v, self.labels:y_v, self.keep_prob_:1.0}
                            loss_v, acc_v = sess.run([self.cost, self.accuracy], feed_dict=feed)
                            vld_acc.append(acc_v)
                            vld_loss.append(loss_v)
                        print("Epoch: {}/{}".format(epoch, self.epochs),
                              "Iteration: {:d}".format(self.iter),
                              "Validation loss: {:6f}".format(np.mean(vld_loss)),
                              "Validation acc: {:.6f}".format(np.mean(vld_acc)))
                        self.validation_acc.append(np.mean(vld_acc))
                        self.validation_loss.append(np.mean(vld_loss))
                    self.iter += 1
            self.saver.save(sess, "checkpoints/model.ckpt")
        self.plot_loss()
        self.plot_acc()
        '''
        #self.test(graph)
        #self.calculate_mean_test(graph)
        prob = self.predict(graph,x_,y_)
        return prob

    def ex(self,graph,s):

        print('data splitting')
        sub_dir = 'train'
        U = util()
        X, y = U.read_data(sub_dir)

        #在训练集中生成验证集，按照1：3
        train_X, vld_X ,train_y, vld_y = train_test_split(X, y, test_size=0.25, random_state=1)
        train_y = U.one_hot(train_y, classes=self.classes)
        vld_y = U.one_hot(vld_y, classes=self.classes)
        print('splitting data finished')

        with tf.Session(graph=graph) as sess:
            oil_cnn.util_placeholder(self, graph)
            oil_cnn.create_network(self, graph)
            print('initializing network finished')

            # 生成checkpoints，Saver
            with graph.as_default():
                self.saver = tf.train.Saver()

            writer = tf.summary.FileWriter(self.log_path, sess.graph)
            tf.summary.scalar('loss', self.cost)
            tf.summary.scalar('accuracy', self.accuracy)
            merged = tf.summary.merge_all()

            #初始化
            init=tf.global_variables_initializer()
            sess.run(init)
            for epoch in range(self.epochs):
                for x,y in U.get_batches(train_X,train_y,batch_size=self.batch_size):
                    feed={self.inputs:x,self.labels:y,self.keep_prob_:self.keep_prob,self.learning_rate_:self.learning_rate}
                    loss,_,acc,prob = sess.run([self.cost, self.optimizer, self.accuracy, self.prob], feed_dict=feed)
                    #print(prob)
                    self.train_acc.append(acc)
                    self.train_loss.append(loss)
                    #print("epoch:"+str(epoch)+"iter:"+str(iter))
                    if self.iter%5 == 0:
                        print("Epoch:{}/{}".format(epoch, self.epochs),
                              "Iteration:{:d}".format(self.iter),
                              "Train loss:{:6f}".format(loss),
                              "Train acc:{:.6f}".format(acc))

                    if self.iter%10 == 0:
                        vld_acc = []
                        vld_loss = []
                        for x_v, y_v in U.get_batches(vld_X, vld_y,batch_size=self.batch_size):
                            feed = {self.inputs:x_v, self.labels:y_v, self.keep_prob_:1.0}
                            loss_v, acc_v = sess.run([self.cost, self.accuracy], feed_dict=feed)
                            vld_acc.append(acc_v)
                            vld_loss.append(loss_v)
                        print("Epoch: {}/{}".format(epoch, self.epochs),
                              "Iteration: {:d}".format(self.iter),
                              "Validation loss: {:6f}".format(np.mean(vld_loss)),
                              "Validation acc: {:.6f}".format(np.mean(vld_acc)))
                        self.validation_acc.append(np.mean(vld_acc))
                        self.validation_loss.append(np.mean(vld_loss))
                    self.iter += 1
            self.saver.save(sess, "checkpoints_test/model.ckpt")
        self.plot_loss(s)
        self.plot_acc(s)

    def predict(self, graph,x_,y_):
        U = util()
        x = []
        y = []
        x.append(x_)
        y.append(y_)
        y = np.array(y)
        y = U.one_hot(y, self.classes)
        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            for i, o in U.get_batches(x, y, batch_size=1):
                feed = {self.inputs:i, self.labels:o, self.keep_prob_:1}
                prob = sess.run(self.prob, feed_dict=feed)
                return prob[0]



    def test(self, graph):
        test_acc = []
        U = util()
        sub_dir = 'test'
        X_, y_ = U.read_data(sub_dir)
        y_ = U.one_hot(y_, self.classes)
        #print(y_)
        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints_test'))
            for x,y in U.get_batches(X_, y_, batch_size=1):
                feed = {self.inputs:x, self.labels:y, self.keep_prob_:1}
                batch_acc,prob = sess.run([self.accuracy, self.prob], feed_dict=feed)
                #print(prob)
                test_acc.append(batch_acc)

            print('Test accuracy:{:.6f}'.format(np.mean(test_acc)))


    def get_mean_test(self, graph):
        test_acc = []
        U = util()
        sub_dir = 'test'
        X_, y_ = U.read_data(sub_dir)
        y_ = U.one_hot(y_, self.classes)
        with tf.Session(graph=graph) as sess:
            self.saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            for x,y in U.get_batches(X_, y_, batch_size=5):
                feed = {self.inputs:x, self.labels:y, self.keep_prob_:1}
                batch_acc = sess.run(self.accuracy, feed_dict=feed)
                test_acc.append(batch_acc)
            return np.mean(test_acc)

    def calculate_mean_test(self,graph):
        test_mean_acc = []
        for i in range(0,10):
            test_mean_acc.append(self.get_mean_test(graph))
            print(str(self.get_mean_test(graph)))
        print("10_mean:"+str(np.mean(test_mean_acc)))


    #Plot loss
    def plot_loss(self,s):
        t = np.arange(self.iter - 1)
        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(self.train_loss), 'r-', t[t % 10 == 0], np.array(self.validation_loss), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/static/image/' + str(s) + 'loss.png')
        #plt.show()

    #Plot accuracies
    def plot_acc(self,s):
        t = np.arange(self.iter - 1)
        plt.figure(figsize=(6, 6))
        plt.plot(t, np.array(self.train_acc), 'r-', t[t % 10 == 0], np.array(self.validation_acc), 'b*')
        plt.xlabel("iteration")
        plt.ylabel("Accuracy")
        plt.legend(['train', 'validation'], loc='upper right')
        plt.savefig('/Users/codergray/PycharmProjects/oil_CNN_version1/oil_CNN/static/image/' + str(s) + 'acc.png')
        #plt.show()

    def judge(self,i):
        if(i == 1):
            return '中水淹层'
        elif(i == 2):
            return '弱水淹层'
        else:
            return '强水淹层'


'''
graph=tf.Graph()
OilCNN=oil_cnn()
OilCNN.exec(graph)
'''










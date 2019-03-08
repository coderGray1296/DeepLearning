import tensorflow as tf
import numpy as np


class CNN_NEW(object):
    def __init__(self, input_size, output_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size], name='input')
        self.input_y = tf.placeholder(tf.float32, [None, output_size], name='output')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')
        print(self.input_x.shape)
        l2_loss = tf.constant(0.0)

        # 为input增加维度input_x.shape拓宽为[batch_size, input_size, input_channels]
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)


        # create convolution + maxpool for every layer

        #第一层卷积层和最大池化层
        filter_shape_1 = [filter_sizes[0], 1, num_filters[0]]
        W_1 = tf.Variable(tf.truncated_normal(filter_shape_1, stddev=0.1), name='W')
        b_1 = tf.Variable(tf.constant(0.1, shape=[num_filters[0]]), name='b')
        cnn_1 = tf.nn.conv1d(
            self.input_x_expanded,
            W_1,
            stride=1,
            padding='VALID',
            name='conv'
        )
        # 卷积结果为[batch_size, input_size - filter_size + 1, num_filters]
        # apply nonlinearity
        h_1 = tf.nn.relu(tf.nn.bias_add(cnn_1, b_1), name='relu')
        print('h_1.shape is:')
        print(h_1.shape)
        # maxpooling layer
        pooled_1 = tf.nn.pool(
            h_1,
            window_shape=[3],
            pooling_type='MAX',
            padding='VALID',
            name='max_pool'
        )
        print('pooled_1.shape is:')
        print(pooled_1.shape)

        #第二层卷积层和最大池化层

        filter_shape_2 = [filter_sizes[1], num_filters[0], num_filters[1]]
        W_2 = tf.Variable(tf.truncated_normal(filter_shape_2, stddev=0.1), name='W')
        b_2 = tf.Variable(tf.constant(0.1, shape=[num_filters[1]]), name='b')

        cnn_2 = tf.nn.conv1d(
            pooled_1,
            W_2,
            stride=1,
            padding='VALID',
            name='conv'
        )
        # 卷积结果为[batch_size, input_size - filter_size + 1, num_filters]
        # apply nonlinearity
        h_2 = tf.nn.relu(tf.nn.bias_add(cnn_2, b_2), name='relu')
        print('h_2.shape is:')
        print(h_2.shape)

        # maxpooling layer
        pooled_2 = tf.nn.pool(
            h_2,
            window_shape=[3],
            pooling_type='MAX',
            padding='VALID',
            name='max_pool'
        )
        print('pooled_2.shape is:')
        print(pooled_2.shape)

        num_filters_total = pooled_2.shape[2]
        # 将列表中的vector在第二个维度上进行整合,shape:[batch_size, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_2, 1)
        # 将结果reshape成二维的数组,shape:[batch_size, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # add dropout layer
        self.drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # output layter and predictions
        with tf.name_scope('output'):
            W = tf.get_variable(
                'W',
                shape=[num_filters_total, output_size],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            b = tf.Variable(tf.constant(0.1, shape=[output_size]), name='b')
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.drop, W, b, name='scores')

        with tf.name_scope('loss'):
            self.losses = abs(self.scores - self.input_y)
            self.loss = tf.reduce_mean(self.losses)

        with tf.name_scope('accuracy'):
            #correct_error = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.accuracy = l2_loss

# cnn = CNN(input_size=5, output_size=1, filter_sizes=[3,4,5], num_filters=10, l2_reg_lambda=0.0)
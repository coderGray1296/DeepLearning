import tensorflow as tf
import numpy as np

class CNN(object):
    def __init__(self, input_size, output_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        #placeholders for input, output, dropout
        self.input_x = tf.placeholder(tf.float32, [None, input_size], name='input')
        self.input_y = tf.placeholder(tf.float32, [None, output_size], name='output')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout')

        l2_loss = tf.constant(0.0)

        #为input增加维度input_x.shape拓宽为[batch_size, input_size, input_channels]
        self.input_x_expanded = tf.expand_dims(self.input_x, -1)

        #create convolution + maxpool for every layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            #convolution layers 卷积核的shape为[filter_size, input_channels, num_filters]
            filter_shape = [filter_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')
            cnn = tf.nn.conv1d(
                self.input_x_expanded,
                W,
                stride=1,
                padding='VALID',
                name='conv'
            )
            #卷积结果为[batch_size, input_size - filter_size + 1, num_filters]

            #apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(cnn, b), name='relu')

            #maxpooling layer
            pooled = tf.nn.pool(
                h,
                window_shape=[input_size - filter_size + 1],
                pooling_type='MAX',
                strides=1,
                padding='VALID',
                name='max_pool'
            )
            #pooled.shape:[batch_size, 1, num_filters]
            pooled_outputs.append(pooled)

        #组合所有的池化之后的特征
        num_filters_total = num_filters * len(filter_sizes)
        #将列表中的vector在第三个维度上进行整合,shape:[batch_size, 1, num_filters_total]
        self.h_pool = tf.concat(pooled_outputs, 2)
        #将结果reshape成二维的数组,shape:[batch_size, num_filters_total]
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        #add dropout layer
        self.drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        #output layter and predictions
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
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_loss + l2_reg_lambda
'''
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.arg_max(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
'''

#cnn = CNN(input_size=5, output_size=1, filter_sizes=[3,4,5], num_filters=10, l2_reg_lambda=0.0)
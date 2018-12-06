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
        input_x_expanded = tf.expand_dims(self.input_x, -1)

        #create convolution + maxpool for every layer
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            #convolution layers 卷积核的shape为[filter_size, input_channels, output_channels]
            filter_shape = [filter_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_size, stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name='b')




cnn = CNN(input_size=5, output_size=1, filter_sizes=[3,4,5], num_filters=10, l2_reg_lambda=0.0)
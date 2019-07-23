import tensorflow as tf
from base.base_model import BaseModel
import numpy as np



"""
    The class implementation 
    of the ProtLigNet model

    
"""

class ProtLigNet(BaseModel):

    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        self.init_saver()
        
    def build_model(self):
    
        graph = tf.Graph()

        with graph.as_default():
            np.random.seed(self.config.seed)
            tf.set_random_seed(self.config.seed)

            after_conv_coord = 21
            
            with tf.variable_scope('input'):
                x = tf.placeholder(tf.float32, 
                                    shape = (None, self.config.input_size, self.config.input_size, self.config.input_size, self.config.num_chls), 
                                    name = 'data_x')
                t = tf.placeholder(tf.float32, shape = (None, self.config.osize), name = "data_affinity")

            with tf.variable_scope('convolution'):
                h_convs = self.convolve3D(x, self.config.conv_channels, conv_patch = self.config.conv_patch, pool_patch = self.config.pool_patch)
                after_conv_coord = h_convs.get_shape()[-2].value
            
            all_var_aft_conv = self.config.conv_channels[-1] * pow(after_conv_coord, 3)

            with tf.variable_scope('f_connected'):
                conv_to_flat = tf.reshape(h_convs, shape = (-1, all_var_aft_conv), name = 'after_conv_flat')

                h_flc = self.feedforward(conv_to_flat, self.config.dense_size, prob = 0.5)

            with tf.variable_scope('output'):
                w = tf.get_variable('w', shape = (self.config.dense_size[-1], self.config.osize),
                                    initializer = tf.truncated_normal_initializer(stddev = (1/(pow(self.config.dense_size[-1], 0.5)))))
                b = tf.get_variable('b', shape = (self.config.osize,), dtype = tf.float32, initializer = tf.constant_initializer(1))

                y = tf.nn.relu(tf.matmul(h_flc, w) + b, name = 'prediction')

            with tf.variable_scope('training'):
                '''
                global_step refers to the number of batches seen by the graph. Every time a batch is provided, 
                the weights are updated in the direction that minimizes the loss. 
                global_step just keeps track of the number of batches seen so far. 
                When it is passed in the minimize() argument list, the variable is increased by one. 
                '''
                
                global_step = tf.get_variable('global_step', shape = (), initializer = tf.constant_initializer(0), trainable = False)
                mse = tf.reduce_mean(tf.pow((y - t), 2), name = 'mse')
                mae = tf.reduce_mean(abs(y - t), name = 'mae')

                with tf.variable_scope('L2_cost'):
                    all_weights = [graph.get_tensor_by_name('convolution/conv_layer_%s/w:0' % index) for index in range(len(self.config.conv_channels))] + [graph.get_tensor_by_name('f_connected/flc_layer%s/w:0' % index) for index in range(len(self.config.dense_size))] + [w]

                    l2 =  self.config.lmbda * tf.reduce_mean([tf.reduce_sum(tf.pow(wi, 2)) for wi in all_weights])

                loss = tf.add(mse, l2, name = "loss")

                optimizer = tf.train.AdamOptimizer(self.config.learning_rate, name = 'optimizer')
                train = optimizer.minimize(loss, global_step = global_step, name = 'train')
        
        self.graph = graph
        # return graph 

    def init_saver(self):
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep = self.config.max_to_keep)

    def feedforward(self, input, neurons_number, prob = 0.5):
        prev = input
        index = 0
        for number in neurons_number:
            output = self.hidden_flc(prev, number, prob, name = 'flc_layer%s' % index)
            prev = output 
            index += 1
        return prev 
    
    @staticmethod 
    def hidden_flc(input, out_chnls, prob, name = "flc_layer"):
        input_channels = input.get_shape()[-1].value

        with tf.variable_scope(name):
            w_shape = (input_channels, out_chnls)
            w = tf.get_variable('w', shape = w_shape, initializer = tf.truncated_normal_initializer(stddev = 0.001))
            b = tf.get_variable('b', shape = (out_chnls,), dtype = tf.float32, initializer = tf.constant_initializer(0.1))
            h = tf.nn.relu(tf.matmul(input,w) + b, name = 'h_result')
            h_drop = tf.nn.dropout(h, prob, name = 'h_dropout')

        return h_drop

    def convolve3D(self, input, channels = [64, 128, 256], conv_patch = 5, pool_patch = 2):
        
        prev = input
        index = 0
        
        for channel in channels:
            result = self.hidden_conv3D(prev, channel, conv_patch, pool_patch, name = "conv_layer_%s" % index)
            prev = result
            index += 1

        return result
    
    @staticmethod 
    def hidden_conv3D(input, out_chnls, conv_patch = 5, pool_patch = 2, name = 'conv_layer'):
        # just the last element in the shape
        input_channels = input.get_shape()[-1].value

        with tf.variable_scope(name):
            w_shape = (conv_patch, conv_patch, conv_patch, input_channels, out_chnls)
            
            w = tf.get_variable('w', shape = w_shape, initializer = tf.truncated_normal_initializer(stddev = 0.001))

            b = tf.get_variable('b', shape = (out_chnls,), dtype = tf.float32, initializer = tf.constant_initializer(0.1))

            conv = tf.nn.conv3d(input, w, strides = [1,1,1,1,1], padding = 'SAME', name = 'conv')

            h = tf.nn.relu(conv + b, name = 'h')

            pool_shape = (1, pool_patch, pool_patch, pool_patch, 1) 
            h_pool = tf.nn.max_pool3d(h, ksize = pool_shape, strides = pool_shape, padding = 'SAME', name = 'h_pool')
        
        return h_pool

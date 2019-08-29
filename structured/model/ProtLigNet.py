import tensorflow as tf
from base.base_model import BaseModel
import numpy as np
import os
from utils.custom_logger import Logger


# the global object logger
logger = Logger(path = os.path.abspath('logs/'), name = "whole_process_log")


class ProtLigNet(BaseModel):
    """The class representation of the project model. It defines the computation graph trough which training process goes. It is extended from BaseModel
    
    :param config: A configuration of the project 
    :type config: JSON
    """

    def __init__(self, config):
        super().__init__(config)
        self.find_model(f"{self.config.model_dir}/model.meta")
        self.init_saver()
        
    def build_model(self):
        """Builds the computational graph of the model and saves the graph as an object's instance attribute"""
    
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
                prob = tf.constant(1.0, name = 'prob_default')
                
                prob_placeholder = tf.placeholder_with_default(prob, shape = (), name = 'keep_prob')
                
                h_flc = self.feedforward(conv_to_flat, self.config.dense_size, prob = prob_placeholder)

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
    
    def find_model(self, path_model):
        """Finds the model in case if it was defined before, otherwise generates a new one
        
        :param path_model: the path to the model
        :type path_model: str
        """
        
        if not os.path.isfile(path_model):
            self.build_model()
            logger.info("The model was created from scratch")
        else:
            _ = tf.train.import_meta_graph(path_model)
            self.graph = tf.get_default_graph()
            logger.info("The model was loaded")           
            
            
    def init_saver(self):
        """Initializes the saver for weights"""
        
        with self.graph.as_default():
            self.saver = tf.train.Saver(max_to_keep = self.config.max_to_keep)

    def feedforward(self, input, neurons_number, prob = 0.5):
        """Propogates through the dense layers
        
        :param input: the flattened weigths after the convolution 
        :type input: tf.TensorArray
        :param neurons_number: the list of nodes number for each dense layer 
        :type neurons_number: list
        :param prob: the probability for dropout, defaults to 0.5
        :type prob: float, optional
        :return: the output after the last dense layer
        :rtype: tf.TensorArray
        """
        
        prev = input
        index = 0
        for number in neurons_number:
            output = self.hidden_flc(prev, number, prob, name = 'flc_layer%s' % index)
            prev = output 
            index += 1
        return prev 
    
    @staticmethod 
    def hidden_flc(input, out_chnls, prob, name = "flc_layer"):
        """Computes weights of a specific dense layer
        
        :param input: the input data
        :type input: tf.TensorArray
        :param out_chnls: the number of nodes in the layer
        :type out_chnls: int
        :param prob: the probability for the dropout
        :type prob: tf.TensorArray
        :param name: the label name of the layer for the tensor graph, defaults to "flc_layer"
        :type name: str, optional
        :return: the output after nonlinear function mapping
        :rtype: tf.TensorArray
        """
        input_channels = input.get_shape()[-1].value

        with tf.variable_scope(name):
            w_shape = (input_channels, out_chnls)
            w = tf.get_variable('w', shape = w_shape, initializer = tf.truncated_normal_initializer(stddev = 0.001))
            b = tf.get_variable('b', shape = (out_chnls,), dtype = tf.float32, initializer = tf.constant_initializer(0.1))
            h = tf.nn.relu(tf.matmul(input,w) + b, name = 'h_result')
            h_drop = tf.nn.dropout(h, prob, name = 'h_dropout')

        return h_drop

    def convolve3D(self, input, channels = [64, 128, 256], conv_patch = 5, pool_patch = 2):
        """Propogates through the convolution layers
        
        :param input: the complex grid of shape (Number of complexes,21,21,21,19)
        :type input: tf.TensorArray
        :param channels: the list of channels for each 3d convolution filter, defaults to [64, 128, 256]
        :type channels: list, optional
        :param conv_patch: the convolution filter width, defaults to 5
        :type conv_patch: int, optional
        :param pool_patch: the pooling filter width, defaults to 2
        :type pool_patch: int, optional
        :return: the output after the convolution layers propogation
        :rtype: tf.TensorArray
        """
        
        prev = input
        index = 0
        
        for channel in channels:
            result = selfKE(prev, channel, conv_patch, pool_patch, name = "conv_layer_%s" % index)
            prev = result
            index += 1

        return result
    
    @staticmethod 
    def hidden_conv3D(input, out_chnls, conv_patch = 5, pool_patch = 2, name = 'conv_layer'):
        """Computes weights of a specific convolution layer
        
        :param input: the input data
        :type input: tf.TensorArray
        :param out_chnls: the number of channels in the convolution layer
        :type out_chnls: int
        :param conv_patch: he convolution filter width, defaults to 5
        :type conv_patch: int, optional
        :param pool_patch: the pooling filter width, defaults to 2
        :type pool_patch: int, optional
        :param name: the label name of the layer for the tensor graph, defaults to 'conv_layer'
        :type name: str, optional
        :return: the output after the convolution
        :rtype: tf.TensorArray
        """
        
        
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

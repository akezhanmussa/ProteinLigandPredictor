import tensorflow as tf
import os
import tensorflow as tf
from data_loader.data_generator import DataGenerator
from datetime import datetime

#     curent_time = 1 #datetime.now().strftime("%d%m%Y_%H:%M:%S")


class GraphLogger:
    """The class representation of the graph summary collection process. It is responsible to records the summaries for weights and errors
    
    :param sess: The main session of the project's graph 
    :type sess: tf.Session 
    :param config: A configuration of the project
    :type config: JSON
    """

    
    def __init__(self, sess, config):
        self.sess = sess
        self.config = config
        
        # the time needed for the event's records 
        self.save_time = datetime.now().strftime("%d:%m:%Y_%H:%M")
        
        # the summary for parameters during the training process
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, f"event_{self.save_time}/train"),
                                                          tf.get_default_graph())
        # the summary for parameters during the validating process
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, f"event_{self.save_time}/test"))

    def summarize(self):
        """Creating summaries for feature importance and other attributes of the graph 
        
        :return: graph summaries of the weights and mse error for training and testing process
        :rtype: a scalar Tensor of type string.
        """
        
        features_map = DataGenerator.get_features_names()
        
        
        # getting the first weights of shape (filter_size, filter_size, filter_size, input_channel, out_channel)
        w_zeroconv =  self.sess.graph.get_tensor_by_name('convolution/conv_layer_0/w:0')
            
        # just splitting the weights by each feature, axis points to the input_channel split
        feature_weights = tf.split(w_zeroconv, w_zeroconv.shape[-2].value, axis = 3)
            # features_importance = tf.reduce_sum(tf.abs(w_zeroconv), reduction_indices = [1,2,4], name = "feature_importance")
        
        summaries = tf.summary.merge((
            tf.summary.histogram('weights', w_zeroconv), 
            *(tf.summary.histogram(f'weights_{feature_name}', value) for feature_name, value in zip(features_map, feature_weights)),
            tf.summary.scalar('mse', self.sess.graph.get_tensor_by_name('training/mse:0')), 
            tf.summary.scalar('mse', self.sess.graph.get_tensor_by_name('training/loss:0'))
        ))

        return summaries
                
                
        
        
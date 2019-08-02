import tensorflow as tf
import os
import tensorflow as tf

class GraphLogger:
    def __init__(self,sess,config):
        self.sess = sess
        self.config = config
        self.train_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"),
                                                          tf.get_default_graph())
        self.test_summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "test"))

    # it can summarize scalars and images.
    def summarize(self):
        """Creating summaries for feature importance
        and other attributes of the graph
        """
        
        # getting the first weights of shape (filter_size, filter_size, filter_size, input_channel, out_channel)
        w_zeroconv =  self.sess.graph.get_tensor_by_name('convolution/conv_layer_0/w:0')
            
        # just splitting the weights by each feature, axis points to the input_channel split
        feature_weights = tf.split(w_zeroconv, w_zeroconv.shape[-2].value, axis = 3)
            # features_importance = tf.reduce_sum(tf.abs(w_zeroconv), reduction_indices = [1,2,4], name = "feature_importance")
        
        summaries = tf.summary.merge((
            tf.summary.histogram('weights', w_zeroconv), 
            *(tf.summary.histogram(f'weights_{index}', value) for index, value in enumerate(feature_weights)),
            tf.summary.scalar('mse', self.sess.graph.get_tensor_by_name('training/mse:0')), 
            tf.summary.scalar('mse', self.sess.graph.get_tensor_by_name('training/loss:0'))
        ))

        return summaries
                
                
        
        
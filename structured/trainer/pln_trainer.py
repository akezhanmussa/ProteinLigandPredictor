from base.base_train import BaseTrain
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import logging

logger = logging.getLogger(name = "whole_process_log")

class ProtLigTrainer(BaseTrain):

    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)
    

    def train(self):
        # tqdm much quickly does the iteration operations

        loop = tqdm(range(self.config.num_iter_per_epoch))
        graph = self.model.graph
        global_step = graph.get_tensor_by_name("training/global_step:0")
        
        prediction = graph.get_tensor_by_name('output/prediction:0')
        self.train_step = graph.get_tensor_by_name('training/train:0')
        self.mse = graph.get_tensor_by_name('training/mse:0')
        self.cost = graph.get_tensor_by_name('training/loss:0')
        w_conv0 = graph.get_tensor_by_name('convolution/conv_layer_0/w:0')

        self.num_epochs = self.config.num_epochs
        self.compare_error = float('inf')

        for epoch in range(self.num_epochs):
            training_mse_err, validation_mse_err = self.train_epoch()
            
            if validation_mse_err < compare_error:
                compare_error = validation_mse_err
                self.model.saver.save(self.sess, 'saved_models/model')

            logger.info("AFTER THE EPOCH {}, the mse_training is {} and mse_validation is {}".format(epoch, training_mse_err, validation_mse_err))

        self.sess.close()

    def train_epoch(self):
        x = self.model.graph.get_tensor_by_name('input/data_x:0')
        t = self.model.graph.get_tensor_by_name('input/data_affinity:0')
        y = self.model.graph.get_tensor_by_name('output/prediction:0')
        
        x_t, y_t = shuffle(range(self.data.dset_sizes['training']), self.data.affinity['training'])
                
        training_mse_err = 0
        validation_mse_err = 0

        for bi, bj in self.data.batches('training'):
            weight = (bj - bi)/self.data.dset_sizes["training"]
            self.sess.run(self.train_step, feed_dict = {x: self.data.g_batch('training', x_t[bi:bj]), t: y_t[bi:bj]})
            train_err = self.sess.run(self.mse, feed_dict = {x: self.data.g_batch('training', x_t[bi:bj]), t: y_t[bi:bj]})
            training_mse_err += weight * train_err

        x_v, y_v = shuffle(range(self.data.dset_sizes['validation']), self.data.affinity['validation'])

        for bi, bj in self.data.batches('validation'):
            weight = (bj - bi)/self.data.dset_sizes["validation"]
            val_err = session.run(self.mse, feed_dict = {x: self.data.g_batch('validation', x_v[bi:bj]), t: y_v[bi:bj]})
            validation_mse_err += weight * val_err
        
        return training_mse_err, validation_mse_err
        


     
                # '''
                #     Add summaries after
                # '''
                # summaries_dict = {
                #     'mse':mse,
                #     'loss':cost,
                #     'weights':weights
                # }
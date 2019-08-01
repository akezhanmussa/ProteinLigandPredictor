from base.base_train import BaseTrain
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import logging
import tensorflow as tf
import pandas as pd
import csv

logger = logging.getLogger(name = "whole_process_log")

class ProtLigTrainer(BaseTrain):

    def __init__(self, sess, model, data, config, logger):
        super().__init__(sess, model, data, config, logger)
    

    def train(self):
        # tqdm much quickly does the iteration operations

        # loop = tqdm(range(self.config.num_iter_per_epoch))
        train_mse_all_error = []
        val_mse_all_error = []
        
        graph = self.model.graph
        global_step = graph.get_tensor_by_name("training/global_step:0")
        
        
        prediction = graph.get_tensor_by_name('output/prediction:0')
        self.train_step = graph.get_tensor_by_name('training/train:0')
        self.mse = graph.get_tensor_by_name('training/mse:0')
        self.cost = graph.get_tensor_by_name('training/loss:0')

        self.num_epochs = self.config.num_epochs
        compare_error = float('inf')
        
        with graph.as_default():
            

            self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.num_epochs):
            training_mse_err, validation_mse_err = self.train_epoch()
            
            if validation_mse_err < compare_error:
                compare_error = validation_mse_err
                self.model.saver.save(self.sess, 'saved_models/model')

            train_mse_all_error.append(training_mse_err)
            val_mse_all_error.append(validation_mse_err)
            logger.info("AFTER THE EPOCH {}, the mse_training is {} and mse_validation is {}".format(epoch, training_mse_err, validation_mse_err))
        
        self.save_to_csv(train_mse_all_error, val_mse_all_error)
        self.sess.close()
        

    # def predict(self):
    #     if os.path.isfile('saved_models/'):
    #             latest_checkpoint = tf.train.latest_checkpoint('saved_models/')
    #             self.saver.restore(self.sess, latest_checkpoint)
        
        
        
    
    
    @staticmethod
    def save_to_csv(train_error, val_error):
        
        with open(os.path.abspath("errors.csv"), 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["train_error", "val_error"])
            writer.writeheader()
            
            for (t_err, v_err) in zip(train_error, val_error):
                writer.writerow({'train_error':t_err, 'val_error': v_err})
            
        logger.info("Errors are recorded in errros.csv")
        
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
            val_err = self.sess.run(self.mse, feed_dict = {x: self.data.g_batch('validation', x_v[bi:bj]), t: y_v[bi:bj]})
            validation_mse_err += weight * val_err
        
        return training_mse_err, validation_mse_err
        


from base.base_train import BaseTrain
import os
from tqdm import tqdm
from sklearn.utils import shuffle
import logging
import tensorflow as tf
import pandas as pd
import csv
from math import sqrt
from datetime import datetime
import math
import numpy as np

logger = logging.getLogger(name = "whole_process_log")

class ProtLigTrainer(BaseTrain):
    """The class representation of the train process. This class is managing all of the other application parts and is responsible for training/testing process.
    It is extended from the BaseTrain class. 
    
    :param sess: The main session of the project's graph 
    :type sess: tf.Session
    :param model: The project model 
    :type model: ProtLigNet 
    :param data: The data object which defines a relation to the input data 
    :type data: DataGenerator 
    :param config: A configuration of the project 
    :type config: JSON
    :param graph_logg: An object is responsible for collecting the graph summaries during the training and testing 
    :type graph_logg: GraphLogger
    """

    def __init__(self, sess, model, data, config, graph_logg):
        super().__init__(sess, model, data, config, graph_logg)
    

    def train(self):   
        """Fuction call for the start of the training process"""  
          
        train_rmse_all_error = []
        val_rmse_all_error = []
        
        graph = self.model.graph
                
        self.train_step = graph.get_tensor_by_name('training/train:0')
        self.mse = graph.get_tensor_by_name('training/mse:0')
        self.cost = graph.get_tensor_by_name('training/loss:0')
        self.summaries = self.graph_logg.summarize()

        self.num_epochs = self.config.num_epochs
        compare_error = float('inf')
        
        with graph.as_default():
            

            self.sess.run(tf.global_variables_initializer())

        for epoch in range(self.num_epochs):
            training_mse_err, validation_mse_err = self.train_epoch(epoch)
            
            if validation_mse_err < compare_error:
                compare_error = validation_mse_err
                self.model.saver.save(self.sess, 'saved_models/model')

            train_rmse_all_error.append(sqrt(training_mse_err))
            val_rmse_all_error.append(sqrt(validation_mse_err))
            logger.info("AFTER THE EPOCH {}, the mse_training is {} and mse_validation is {}".format(epoch, sqrt(training_mse_err), sqrt(validation_mse_err)))
        
        self.graph_logg.train_summary_writer.close()
        self.graph_logg.test_summary_writer.close()
        self.record_errors(train_rmse_all_error, val_rmse_all_error, self.config.error_dir, self.graph_logg.save_time)
        self.sess.close()
        
        
    def train_epoch(self, epoch_num = 0):
        """Trains the model during one epoch
        
        :param epoch_num: the current epoch index of training
        :type epoch_num: int, optional
        :return: the training and validation errors on this particular epoch
        :rtype: float, float
        """
        x = self.model.graph.get_tensor_by_name('input/data_x:0')
        t = self.model.graph.get_tensor_by_name('input/data_affinity:0')
        y = self.model.graph.get_tensor_by_name('output/prediction:0')
        keep_prob = self.model.graph.get_tensor_by_name('f_connected/keep_prob:0')
        
        x_t, y_t = shuffle(range(self.data.dset_sizes['training']), self.data.affinity['training'])
          
        training_mse_err = 0
        validation_mse_err = 0

        for bi, bj in self.data.batches('training'):
            weight = (bj - bi)/self.data.dset_sizes["training"]
            self.sess.run(self.train_step, feed_dict = {x: self.data.g_batch('training', x_t[bi:bj]), t: y_t[bi:bj], keep_prob: 0.5})
            train_err = self.sess.run(self.mse, feed_dict = {x: self.data.g_batch('training', x_t[bi:bj]), t: y_t[bi:bj]})
            training_mse_err += weight * train_err

        train_summaries = self.sess.run(self.summaries, feed_dict = {
            x:self.data.g_batch('training', x_t[:self.config.batch_size]),
            t:y_t[:self.config.batch_size], keep_prob:1.0})
        
        self.graph_logg.train_summary_writer.add_summary(train_summaries, epoch_num)
        
        x_v, y_v = shuffle(range(self.data.dset_sizes['validation']), self.data.affinity['validation'])

        
        for bi, bj in self.data.batches('validation'):
            weight = (bj - bi)/self.data.dset_sizes["validation"]
            val_err = self.sess.run(self.mse, feed_dict = {x: self.data.g_batch('validation', x_v[bi:bj]), t: y_v[bi:bj]})
            validation_mse_err += weight * val_err
            
        test_summaries = self.sess.run(self.summaries, feed_dict = {
                x:self.data.g_batch('validation', x_v[:self.config.batch_size]),
                t:y_v[:self.config.batch_size], keep_prob:1.0})
        
        self.graph_logg.test_summary_writer.add_summary(test_summaries, epoch_num)

        return training_mse_err, validation_mse_err
        

    def predict(self):
        """Function call for the start of the prediction process"""
        
        y = self.model.graph.get_tensor_by_name('output/prediction:0')
        x = self.model.graph.get_tensor_by_name('input/data_x:0')
        t = self.model.graph.get_tensor_by_name('input/data_affinity:0')
        mse = self.model.graph.get_tensor_by_name('training/mse:0')

        logger.info("The prediction is started")
        
        predictions = []
        real_values = []
        data_codes = []
    
        
        if os.path.isdir(os.path.abspath('saved_models/')):
            latest_checkpoint = tf.train.latest_checkpoint('saved_models/')
            self.model.saver.restore(self.sess, latest_checkpoint)
            
            print("Shape of the testing set", (self.data.dset_sizes['testing']))
            # testing_mse_err = 0
            
            for bi, bj in self.data.batches('testing'):
                weight = (bj - bi)/self.data.dset_sizes["testing"]
                prediction = self.sess.run(y, feed_dict= {x: self.data.g_batch('testing', range(bi,bj))})
                
                predictions.append(self.convert_to_mol(prediction[0]))
                data_codes.append(self.data.id_s['testing'][bi:bj][0])
                

            
            predictions = np.array(predictions)
            data_codes = np.array(data_codes)
            
            predictions = predictions.reshape((predictions.shape[0],))
            self.record_predictions([(predictions,"predictions"), (real_values,"real_values"), (data_codes,"data_codes")], name = self.config.specific_test_name[0:-4])
        else:
            logger.info("The model did not exist, please upload model meta files")
        
                
    @staticmethod
    def convert_to_mol(affinity):
        """Converts the -log10 valus to micromoles
        
        :param affinity: the -log10 affinity represenation value
        :type affinity: float
        :return: the micromol representation of affinity value
        :rtype: float
        """

        uPower = 10**6
        converted = 1/math.pow(10, affinity)
        converted *= uPower
        return converted
    
    @staticmethod 
    def record_predictions(data, name = "390"):
        """Records the dict of predictions as a csv file 
        
        :param data: the list of different data types (e.g: predictions, real_values or data_codes)
        :type data: dict
        :param name: the name of the csv file
        :type name: str, optional
        """
        
        input_dict = {}
        columns = []
        
        for data_column in data:
            if not len(data_column[0]) == 0:
                input_dict[data_column[1]] = data_column[0] 
                columns.append(data_column[1])
                
        data_set = pd.DataFrame(input_dict, columns=columns)
        predictions_folder = "predictions_on_test_cases"
        data_set.to_csv(os.path.abspath(f"logs/{predictions_folder}/{name}_predictions.csv"))
        print("Predictions are recorded")
        
     
    @staticmethod
    def record_errors(train_error, val_error, path, time):
        """Records the trainining and validation errors on the specific time range as a csv file
        
        :param train_error: list of training errors
        :type train_error: list
        :param val_error: list of validation errors
        :type val_error: list
        :param path: the path to the csv file
        :type path: str
        :param time: the time of recordings
        :type time: str 
        """
                
        with open(f"{path}/errors_{time}.csv", 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["train_error", "val_error"])
            writer.writeheader()
            
            for (t_err, v_err) in zip(train_error, val_error):
                writer.writerow({'train_error':t_err, 'val_error': v_err})
            
        logger.info("Errors are recorded in errros.csv")
        
    
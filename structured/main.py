import os
from utils.config import get_config
from model.ProtLigNet import ProtLigNet
from data_loader.data_generator import DataGenerator
from utils.custom_logger import Logger
from utils.graph_logger import GraphLogger
from trainer.pln_trainer import ProtLigTrainer
import tensorflow as tf


def main():

    # Reading the config from JSON
    config, _ = get_config(os.path.abspath('configs/pln_config.json'))
    
    # Defining the model
    model = ProtLigNet(config)
    
    # Defining the data generator
    data = DataGenerator(config)
    
    # Some setup for the gpu usage
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    
    # The session for running training process
    sess = tf.Session(graph = model.graph, config = gpu_config)
    # the custom logger for recording the events during training
    logger = GraphLogger(sess, config)
    # trainer, directly interacting with trained weights
    trainer = ProtLigTrainer(sess, model, data, config, logger)
    
    trainer.predict()
    #trainer.train()

if __name__ == "__main__":
    main()

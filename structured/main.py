import os
from utils.config import get_config
from model.ProtLigNet import ProtLigNet
from data_loader.data_generator import DataGenerator
from utils.custom_logger import Logger
from utils.graph_logger import GraphLogger
from trainer.pln_trainer import ProtLigTrainer
import tensorflow as tf


def main():
    config, _ = get_config(os.path.abspath('configs/pln_config.json'))
    
    model = ProtLigNet(config)
    data = DataGenerator(config)
    
    """
        Some setup for the gpu usage
    """
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(graph = model.graph, config = gpu_config)
    logger = GraphLogger(sess, config)
    trainer = ProtLigTrainer(sess, model, data, config, logger)
    trainer.predict()
    

if __name__ == "__main__":
    main()
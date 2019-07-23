import os
os.chdir('/home/mussaa/Downloads/AkeFiles/Files/project_ver2/new_version/ProteinLigandPredictor/structured')
print(os.getcwd())
from utils.config import get_config
from model.ProtLigNet import ProtLigNet
from data_loader.data_generator import DataGenerator
from utils.custom_logger import Logger
from trainer.pln_trainer import ProtLigTrainer


def main():
    config_path = get_config(os.pardir(os.path.abspath("./") + 'pln_config.json'))
    config, _ = get_config(config_path)
    data = DataGenerator(config)
    model = ProtLigNet(config)
    """
        Some setup for the gpu usage
    """
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(graph = model.graph, config = config)
    logger = Logger(sess, config)

    trainer = ProtLigTrainer(sess, model, data, config, logger)
    trainer.train()
    

if __name__ == "__main__":
    main()
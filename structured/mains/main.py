
from utils.config import get_config
from model.ProtLigNet import ProtLigNet
from data_loader.data_generator import DataGenerator
from utils.custom_logger import Logger
import os


def main():
    config_path = get_config(os.pardir(os.path.abspath("./") + 'pln_config.json'))
    config, _ = get_config(config_path)
    data_generator = DataGenerator(config)
    model = ProtLigNet(config)
    

if __name__ == "__main__":
    main()
import json 
from bunch import Bunch
import os 

def get_config(json_file):

    with open(json_file, 'r') as c_file:
        config_dict = json.load(c_file)

    config = Bunch(config_dict)

    return config, config_dict














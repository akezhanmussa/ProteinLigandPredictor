import json 
from bunch import Bunch
import os 

def get_config(json_file):
    """Returns the parsed version of the json file
    
    :param json_file: The json file to be parsedf
    :type json_file: JSON
    :return: the dict version
    """

    with open(json_file, 'r') as c_file:
        config_dict = json.load(c_file)

    config = Bunch(config_dict)

    return config, config_dict














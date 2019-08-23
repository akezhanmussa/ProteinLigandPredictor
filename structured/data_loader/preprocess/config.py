import json

def get_config(json_file):
    """Reading attributes to the dict 
    from the config
    
    param json_file: the config file
    
    return config_dict: the dict representation of config
    """

    with open(json_file, 'r') as c_file:
        config_dict = json.load(c_file)

    return config_dict
import pandas as pd
import numpy as np
import csv
import os
import re
import math



unit_dict = {
    'm':10**-3,
    'u':10**-6,
    'p':10**-12,
    'n':10**-9,
    'f':10**-15
}


def find_ratio(path_one):
    """Finding the ratio of real logarithm 
    to the one provided by the pdb bind database
    
    
    param path_one: the path to pdb_bind db csv    
    """
    
    data_one = pd.read_csv(path_one, error_bad_lines= False)
    print(data_one.head())
    print("*"*20)
    print(data_one.columns)
    data_one['true_log'] = (-1) * np.log10(pd.to_numeric(data_one["Kd/Ki"]))
    print(data_one.head())
    data_one['ratio_log'] = data_one['-logKd/Ki']/data_one['true_log']
    
    data_one.to_csv("INDEX_filtered_affinity.csv")
    

def get_unit(unit):
    global unit_dict
    return unit_dict[unit]


def prepare_affinity_data(path):
    path_to_affinity_csv = os.path.abspath("INDEX_affinity.csv")
    
    with open(path, 'r') as f:
        rows = f.readlines()
        index = 0
        with open(path_to_affinity_csv, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["index", "pdbid", "-logKd/Ki", "Kd/Ki"])
            writer.writeheader()

            for row in rows:
                splitted_row = row.split()
                if (splitted_row[0] == "#"):
                    continue
                else:
                    try:    
                        in_power = get_unit(splitted_row[4][-2:-1])
                    except:
                        raise Exception(f"The metric does not exist {splitted_row[4][-2:-1]}")
                    
                    filtered_number = re.sub(r'(IC50)|[^0-9\.]', '', splitted_row[4])
                    filtered_number = float(filtered_number) * in_power     
                    writer.writerow({"index":index, "pdbid":splitted_row[0], "-logKd/Ki":splitted_row[3], "Kd/Ki":filtered_number})
                    index += 1

  
    
if __name__ == "__main__":
    prepare_affinity_data("INDEX_general_PL_data.2018")
    find_ratio(os.path.abspath("INDEX_affinity.csv"))
    
    
    
    
    

        
    
    
    
    
    
    
    
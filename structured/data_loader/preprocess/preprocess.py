import argparse
import os
# import pandas as pd
import h5py
from constant import Global_var
from helper import extract_features
import numpy as np
# module which contains built-in functions 
import builtins
import csv
from custom_logger import Logger
import json 
import pandas as pd
import math
from helper import is_file_empty
import shutil
from folder_functions import FolderFunctions

miss_container = {}
ignore_list = []


'''
    Adding the logger object

    levelname -> the type of logger 
    
'''

logger = Logger(path = os.path.abspath("logs"), warning_to_file = True)


"""

    Method_2: how to read affinity data 

"""
def get_affinity(file_path, affinity_db, is_csv = False):
    """Get the affinity json file
     of specific folder of complexes
     
    :param file_path: path to json files, does not matter whether it exists or not
    :param affinity_db: path to the file where affinity can be extracted
    :param is_csv: set True if affinity_db is of csv format 
    
    :return: the affinities of the folder of complexes
    """
    
    
    # in case if affinity json does not exist
    if not os.path.isfile(file_path):
        affinity = {}
        # in case if affinity is not of csv format
        if not is_csv:
            index = 0
            with open(affinity_db, 'r') as f:
                for line in f:
                    if (line[0] != "#"):
                        splitted = line.split()
                        affinity[splitted[0]] = splitted[3]
                    else:
                        index = -1
                    index += 1
        else:
            data = pd.read_csv(affinity_db)
            metric_dict = Global_var.METRIC_CONVERT.value
            
            # deleting the rows with Ka metric affinity 
            data.drop(data.loc[data["Metric"].isin(["Ka"])].index, axis = 0, inplace = True)
            
            complex_code = data["Representative"].apply(lambda x: x.lower())
            
            # converting unit to the corresponding power
            data["Unit"] = data["Unit"].apply(lambda x: metric_dict[x])
            
            complex_affinity = pd.to_numeric(data["Num"]) * pd.to_numeric(data["Unit"]) 
            
            # applying to each affinity, the -log10 to imitate pdbbind bind data set
            complex_affinity = complex_affinity.apply(lambda x: (-1) * math.log10(x))
            
            affinity = pd.Series(complex_affinity.values, index = complex_code).to_dict()

        with open(file_path, 'w') as f:
                json.dump(affinity, f)
    
    else:
        with open(file_path, 'r') as f:
            affinity_data = json.load(f)
        return affinity_data

"""

    Protonoted and Charged the molecules 
    with chimera commands

"""

def charge_protonate(path_to_db, affinity_data = None, affinity_required = False):
    
    complex_set = os.listdir(path_to_db)
    print(complex_set)
    
    count = 0

    for _, molecule_name in enumerate(complex_set): 
        
        # dicsard all of the complexes without affinites
        if affinity_required:    
            try:
                assert affinity_data[molecule_name]
            except:
                continue
             
        if check_molecule_existence(path_to_db, molecule_name): continue   
        
        path_to_complex = "%s/%s/%s_protein.pdb" % (path_to_db, molecule_name, molecule_name)
        os.system("chimera --nogui {} some.py".format(path_to_complex))

        if not check_molecule_existence(path_to_db, molecule_name):
            logger.warning("{} was not read by chimera, it was excluded".format(molecule_name))
            ignore_list.append(molecule_name)

        count += 1
        print("==============",molecule_name,"================")
        print("==============",count,"================")

    if (len(ignore_list) != 0):
        logger.warning("TOTAL NUMBER OF EXCLUDED COMPLEXES {}".format(len(ignore_list)))

    logger.info("PROTONATION DONE, TOTAL NUMBER OF COMPLEXES {}".format(count))

def check_molecule_existence(path_to_db, molecule_name):
    possible_path_mol2 = "%s/%s/%s.mol2" % (path_to_db, molecule_name, molecule_name)
    possible_path_json = "%s/%s/%s.mol2" % (path_to_db, molecule_name, molecule_name) 
    
    if os.path.isfile(possible_path_mol2) and os.path.isfile(possible_path_json): 
        if is_file_empty(possible_path_mol2) or is_file_empty(possible_path_json):
            return False
        else:
            return True
    else:
        return False
            
            
def get_ignore_list(path):   
    """Read the logg file for finding the excluded complexes

    :param path: path to logg file
    :return: the array of excluded complexes

    """
    global ignore_list

    if (len(ignore_list) == 0):
        with open(path, 'r') as logg_file:
            for line in logg_file:
                line = line.split()
                ignore_list.append(line[0][-4:])        
    return ignore_list
    

def prepare_hdf(db_path, name = "data.hdf",  missed_name = 'missed_atoms', batch_num = 50, affinity_data = None, affinity_required = True):
    path_hdf = os.path.abspath("preprocessed_data/" + name)
    global ignore_list
    complex_set = os.listdir(db_path)
    
    ignore_list = ignore_list if len(ignore_list) > 0 else []#get_ignore_list(os.path.abspath("logs/logger.logg"))
    print(ignore_list)
    processed_batch = []
    
    with h5py.File(path_hdf, 'w') as f:
        index = 0

        for _, row in enumerate(complex_set):
            
            if affinity_required:
                try:
                    assert affinity_data[row]
                except:
                    print(f"COMPLEX {row} DOES NOT HAVE CHARGE")
                    continue
                
            if row in ignore_list: continue 
        
            ligands_coords, ligands_features, relative_central_mass = extract_features(miss_container, row, molcode = 1, db_path=db_path)
            
            if not is_valid(ligands_coords, ligands_features): continue
                        
            pocket_coords, pocket_features = extract_features(miss_container, row, molcode = -1, db_path=db_path, relative_central_mass=relative_central_mass)
            
            if not is_valid(pocket_coords, pocket_features): continue

            if pocket_coords.shape[0] == 0 and pocket_features.shape[0] == 0: continue

            center = ligands_coords.mean(axis = 0)
            '''
                to make center as (0,0,0)
                subtract each coordinate by 
                their mean  
            '''
      
            ligands_coords -= center
            pocket_coords -= center

            '''
                merge rows of ligands_coords
                with pocket_coords

                then add columns for rows of 
                ligands and pockets with features
            
            '''
            complex_data = np.concatenate((np.concatenate((ligands_coords, pocket_coords)), np.concatenate((ligands_features, pocket_features))), axis = 1)
            some_shape = complex_data.shape
            data_set = f.create_dataset(row, data = np.concatenate((np.concatenate((ligands_coords, pocket_coords)), np.concatenate((ligands_features, pocket_features))), axis = 1), shape = some_shape, dtype = "float32", compression = 'lzf')
            
            if affinity_required:
                data_set.attrs["affinity"] = affinity_data[row]
            processed_batch.append(row)
            
            if (len(processed_batch) == batch_num):
                logger.info(f"DONE WITH {len(processed_batch)} COMPLEXES")
                processed_batch = []

            index += 1

        if (not len(processed_batch) == 0):
            logger.info(f"DONE WITH LAST {len(processed_batch)} COMPLEXES")

    if (len(miss_container) != 0):
        logger.warning("The complexes with missed atomos were recorded in the missed_atoms.csv")
        title_miss = ["Complex", "Missed Atoms"]        
        with open(f'logs/{name}.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames = title_miss)
            writer.writeheader()
            for key,value in miss_container.items():
                writer.writerow({'Complex':key, 'Missed Atoms':value})


def is_valid(coords, features):
    if isinstance(coords, np.ndarray) and isinstance(features, np.ndarray):
        if coords.shape == (1,)  or features.shape == (1,):
            return False
        return True
    return False


if __name__ == "__main__":
    # divide_by_folders()
    # convert_folders_to_mol2()
    
    
    
    
    
    
    # convert_folders_to_mol2()
    
    
    #affinity_data = get_affinity(file_path = "affinity-rec-lig.json", affinity_db = os.path.abspath(f"pdbbind_data_2018/index_rec_lig/formatted_binding.csv"), is_csv = True)
    
    
    
    
    #prepare_hdf(affinity_data = affinity_data, db_path = db_path, name = "rec-log.hdf" )
    
    
    
    # affinity_path = os.path.abspath("pdbbind_data_2018/index_core_set/2013_core_data.lst")
    # prepare_hdf(file_path = "affinity-core.json", affinity_db = affinity_path, db_path = db_path, name = "core_set.hdf")
    
    
    folder_name = "specific_complexes"
    db_path = f"{Global_var.DB.value}/{folder_name}"
    FolderFunctions.convert_folders_to_mol2(path_to_folder = db_path, specific_ending = "_ligand.pdb")
    charge_protonate(db_path, affinity_required = False)
    prepare_hdf(db_path = db_path, name = f"{folder_name}.hdf", affinity_required = False)

    # split_many_ligands()
    # merge_ligand_complexes_with_proteins()

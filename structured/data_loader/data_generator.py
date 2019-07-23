import numpy as np
import sys
import h5py
import os
import logging
# sys.path.append("../utils")
from utils.custom_logger import Logger
from utils.constant import Constants
from utils.config import get_config

from sklearn.utils import shuffle


'''
    Attributes of config:
        datasets
        data_type
        data_path
        data_name
        batch_size
'''

logger = Logger(path = os.path.abspath('logs/'), name = "whole_process_log")

class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.split_data(self.config.data_name)
        self.fill_data()

    def split_data(self, file_name):
        
        hdf_path = self.config.data_path
        hdf_file = hdf_path + '/' + file_name
        path_training = hdf_path + '/' +  "training.hdf"
        path_validation = hdf_path + '/' + "validation.hdf"
        

        if not os.path.isfile(path_training) and not os.path.isfile(path_validation):
            logger.info("the splitting has started")
            with h5py.File(path_training, 'w') as t, \
                h5py.File(path_validation, 'w') as v:
                with h5py.File(hdf_file, 'r') as f:
                    shuffled_keys = shuffle(list(f.keys()), random_state = 14)
                    size = len(shuffled_keys)
                    
                    for pdb_id in shuffled_keys[:int(size*self.config.split_size)]:
                        ds = t.create_dataset(pdb_id, data = f[pdb_id])
                        ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
                    
                    for pdb_id in shuffled_keys[int(size*self.config.split_size):]:
                        ds = v.create_dataset(pdb_id, data = f[pdb_id])
                        ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
            logger.info("The splitting is done")
        else:
            logger.info("The splitting was done before")

    def fill_data(self):

        self.id_s = {}
        self.affinity = {}
        self.coords = {}
        self.features = {}

        logger.info("The filling data is started")
        
        for dictionary in [self.id_s, self.affinity, self.coords, self.features]:
            for dataset in self.config.datasets:
                dictionary[dataset] = []
        

        for dataset in self.config.datasets:

            file_path = os.path.abspath("%s/%s.hdf" % (self.config.data_path, dataset))

            if not os.path.isfile(file_path): continue

            with h5py.File(file_path, 'r') as f:
                for pdb_id in f:
                    one_set = f[pdb_id]
                    self.coords[dataset].append(one_set[:, :3])
                    self.features[dataset].append(one_set[:, 3:])
                    self.affinity[dataset].append(one_set.attrs['affinity'])
                    self.id_s[dataset].append(pdb_id)
            
            self.id_s[dataset] = np.array(self.id_s[dataset])
            self.affinity[dataset] = np.reshape(self.affinity[dataset], (-1, 1))
        
        features_set = self.get_features_names()
        features_map = {name:i for i,name in enumerate(features_set)}
        
        charges = []
        for feature_data in self.features["training"]:
            '''
                Extraction of charges of each molecule, 
                just pointing the column number

                notation -> ..., columnIndex
                extracts the whole specific column by specifying 
                the col. index
            '''
            charges.append(feature_data[..., features_map["partialcharge"]])
        
        '''
            flatten the array of each molec. char vector to one vector
        '''
        
        charges = np.concatenate([charge.flatten() for charge in charges])
        self.mean_charge = charges.mean()
        self.std_charge = charges.std()


        '''
            indexes where the molcode equals to one and
            minus one
        '''
        # temp_batch = g_batch(indices = range(50))
        # index_one = [[i[0]] for i in np.where(temp_batch[..., features_map['molcode']] == 1.0)]

        self.dset_sizes = {dataset: len(self.affinity[dataset]) for dataset in self.config.datasets}
        self.num_batches = {dataset: (size//self.config.batch_size) for dataset, size in self.dset_sizes.items()}

        logger.info("the filling is done")
    
    def batches(self, set_name):
        for b in range(self.num_batches[set_name]):
            bi = b * 10
            bj = (b + 1)*10
            if b == self.num_batches[set_name] - 1:
                bj = self.dset_sizes[set_name]
            yield bi, bj

    def g_batch(dataset = 'training', indices = range(0,10), rotation = 0):
        global coords, features, std, features_map
        x = []
        for index in indices:
            '''
                coordinates and features of one 
                complex
            '''

            coords_index = coords[dataset][index]
            
            feature_index = features[dataset][index]
            x.append(to_box(coords_index, feature_index))
            
        x = np.vstack(x)
        x[..., features_map['partialcharge']] /= std
        return x 
    
    @staticmethod
    def get_features_names():
        atom_classes = Constants.atom_classes.value
        '''
            the order of features in the data set
        '''
        features = []
        
        '''
            added the atom names 
        '''
        for index, (atom_num, name) in enumerate(atom_classes):
            features.append(name)

        features += Constants.NAMED_PROPS.value
        features.append('molcode')
        features += Constants.smarts_labels.value

        return features
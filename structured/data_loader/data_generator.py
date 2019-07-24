import numpy as np
import sys
from math import ceil
# sys.path.append("../utils")
from utils.custom_logger import Logger
from utils.constant import Constants
from utils.config import get_config
from sklearn.utils import shuffle
import h5py
import os
import logging
from math import pi, sin, cos

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
        """Splitting the whole data set to train and validation set
        
        Split the data using the following parameters.
        
        :param file_name: The name of the data set
        """        
        
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
            
    
    def apply_rotation(self, x_angle, y_angle, z_angle):
        
        rot_x = np.array([[1, 0, 0],
                          [0, cos(x_angle), -sin(x_angle)],
                          [0, sin(x_angle),cos(x_angle)])
        rot_y = np.array([[cos(y_angle, 0, sin(y_angle))],
                          [0, 1, 0],
                          [-sin(y_angle), 0, cos(y_angle)])
        rot_z = np.array([[cos(z_angle),-sin(z_angle),0],
                          [sin(z_angle),cos(z_angle),0],
                          [0,0,1]]])
        
        mul_x_y = np.dot(rot_x, rot_y)
        result = np.dot(mul_x_y, rot_z)
        
        return result

    
            
    

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
        self.features_map = {name:i for i,name in enumerate(features_set)}
        
        charges = []
        for feature_data in self.features["training"]:
            '''
                Extraction of charges of each molecule, 
                just pointing the column number

                notation -> ..., columnIndex
                extracts the whole specific column by specifying 
                the col. index
            '''
            charges.append(feature_data[..., self.features_map["partialcharge"]])
        
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

    
    
    def g_batch(self, dataset = 'training', indices = range(0,10), rotation = 0):
        x = []
        for index in indices:
            '''
                coordinates and features of one 
                complex
            '''

            coords_index = self.coords[dataset][index]
            
            feature_index = self.features[dataset][index]
            x.append(self.to_box(coords_index, feature_index))
            
        x = np.vstack(x)
        x[..., self.features_map['partialcharge']] /= self.std_charge
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
    
    @staticmethod
    def to_box(coords, features, grid_resolution = 1.0, max_dist = 10.0):
        '''
            representing the coordinates as 3d grid with 21 Angstons diameter
            for one molecular complex
        '''

        coords_num = len(coords)
        f_shape = features.shape
        features_num = f_shape[1]

        box_size = ceil(2*max_dist + 1)

        '''
            change coordinates of atoms to be relative to the center of the box
        '''

        grid_coords = (coords + max_dist) / grid_resolution 


        # copies the arrays and casts its elements as integer
        grid_coords = grid_coords.round().astype(int)

        # detect which atoms are in the box 
        # axis = 1 indicates the values in the rows would be checked
        # since the shape of the coords - (N, 3)
        inside_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis = 1)

        grid = np.zeros((1, box_size, box_size, box_size, features_num), dtype = np.float32)
        
        for (x, y, z), f in zip(grid_coords[inside_box], features[inside_box]):
            grid[0, x, y, z] += f

        return grid
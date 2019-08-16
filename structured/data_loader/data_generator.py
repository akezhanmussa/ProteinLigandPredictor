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
from math import pi, sin, cos, sqrt
from itertools import combinations, product, permutations
from data_loader.pdb_reader import Options
from collections import defaultdict

np.seterr(invalid = 'raise')
'''
    Attributes of config:
        datasets
        data_type
        data_path
        data_name
        batch_size
'''

global_index = 0

logger = logging.getLogger(name = "whole_process_log")

class DataGenerator:
    def __init__(self, config):
        self.config = config
        
        self.split_data(self.config.data_name)
        
        # save the rotaion matrices
        self.fill_rotations()
        
        # filling the 
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
        path_testing = hdf_path + '/' + "testing.hdf"
        # hdf_core_file = hdf_path + '/' + "core_set.hdf"
        hdf_core_file = hdf_path + '/' + "rec-log.hdf"

        
        # benchmark set
        # core_set = Options.read_data(os.path.abspath("data_loader/core_pdbbind2013.ids"))
        core_set = []
        
        # to be sure not to add complexes to testing.hdf twice
        excluded_complexes = defaultdict(lambda: False)

        if not os.path.isfile(path_training) or not os.path.isfile(path_validation) or not os.path.isfile(path_testing):
            logger.info("the splitting has started")
            with h5py.File(path_training, 'w') as t, \
                h5py.File(path_validation, 'w') as v, \
                h5py.File(path_testing, 'w') as k:
                        with h5py.File(hdf_file, 'r') as f:
                            shuffled_keys = shuffle(list(f.keys()), random_state = 14)
                            size = len(shuffled_keys)
                            
                            for pdb_id in shuffled_keys[:int(size*self.config.split_size)]:
                                
                                if pdb_id in core_set:
                                    ds = k.create_dataset(pdb_id, data = f[pdb_id])
                                    ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
                                    excluded_complexes[pdb_id] = True
                                else:
                                    ds = t.create_dataset(pdb_id, data = f[pdb_id])
                                    ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
                            
                            for pdb_id in shuffled_keys[int(size*self.config.split_size):]:
                                
                                if pdb_id in core_set:
                                    ds = k.create_dataset(pdb_id, data = f[pdb_id])
                                    ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
                                    excluded_complexes[pdb_id] = True
                                else:
                                    ds = v.create_dataset(pdb_id, data = f[pdb_id])
                                    ds.attrs['affinity'] = f[pdb_id].attrs["affinity"]
                        
                        with h5py.File(hdf_core_file, 'r') as m:
                            for pdb_id in m.keys():
                                if excluded_complexes[pdb_id] == False:
                                    ds = k.create_dataset(pdb_id, data = m[pdb_id])
                                    ds.attrs['affinity'] = m[pdb_id].attrs["affinity"]
                                                            
            logger.info("The splitting is done")
        else:
            logger.info("The splitting was done before")
    
    def fill_rotations(self):
        '''Predefine the rotation matrices
        for data augmentation of the complexes
        '''
        
        # The first rotation matrix is just the identity mapping
        self.rotation_matrices = [self.get_rotation_matrix(np.array([1,1,1]), 0)]
            
        # 27 different rotations axices, with 100 angles for each
        # matrix contains 2700 rotations 
        for (x,y,z) in product([0,-1,1], repeat = 3):
            for n in range(1, 120):
                axis = np.array([x,y,z])
                angle = n * (pi/60)
                self.rotation_matrices.append(self.get_rotation_matrix(axis, angle))
   
        logger.info("The rotations matrices are processed")
                
    @staticmethod
    def get_rotation_matrix(axis, angle):
        
        '''Rotation matrix for any angle 
        around any axis
        
        angle should be in range from [-1, 1] 
        '''
        
        # normalize the units of the rotation axis
        
        try:
            if not np.dot(axis,axis) == 0:
                axis = axis / sqrt(np.dot(axis, axis))
        except FloatingPointError as err:
            print(axis)
            print(type(axis))
            print(str(err))

        u1, u2, u3 = axis
        
        c = cos(angle)
        s = sin(angle)
        v = 1 - c
        
        rotation_matrix = np.array([[u1*u1*v + c,u1*u2*v - u3*s, u1*u3*v + u2*s],
                                    [u1*u2*v + u3*s, u2*u2*v + c, u2*u3*v - u1*s],
                                    [u1*u3*v - u2*s, u2*u3*v + u1*s, u3*u3*v + c]])
        return rotation_matrix
    
    @staticmethod
    def rot_random(dataset, indices):
        
        pair = []
        
        if dataset == 'validation' or dataset == 'testing':
            return [0]
        elif dataset == 'training':
            while True:
                x = np.random.choice(indices, 20)
                x = np.append(x, 0)
                unique_three = set(x)
            
                if len(unique_three) >= 6:
                    break
            
            for index, elem in enumerate(unique_three):
                if index == 6:
                    break
                pair.append(elem)
            
        if len(pair) == 0:
            logger.info("The wrong dataset name, check the name")
        
        return pair
   
    def fill_data(self):

        self.id_s = {}
        self.affinity = {}
        self.coords = {}
        self.features = {}
        self.charges = {}
        
        logger.info("The filling data has started")
        
        for dictionary in [self.id_s, self.affinity, self.coords, self.features,self.charges]:
            for dataset in self.config.datasets:
                dictionary[dataset] = []
        
        features_set = self.get_features_names()
        self.features_map = {name:i for i,name in enumerate(features_set)}
        
        for dataset in self.config.datasets:
        
            file_path = os.path.abspath("%s/%s.hdf" % (self.config.data_path, dataset))
            if not os.path.isfile(file_path): continue
            
            with h5py.File(file_path, 'r') as f:
                for pdb_id in f:
                    rotation_index = self.rot_random(dataset, len(self.rotation_matrices))
                   
                    for idx in rotation_index:
                        one_set = f[pdb_id]
                        self.coords[dataset].append(np.dot(one_set[:, :3],self.rotation_matrices[idx]))
                        self.features[dataset].append(one_set[:, 3:])
                        self.affinity[dataset].append(one_set.attrs['affinity'])
                        self.id_s[dataset].append(pdb_id)

            self.id_s[dataset] = np.array(self.id_s[dataset])
            self.affinity[dataset] = np.reshape(self.affinity[dataset], (-1, 1))        

            
            for feature_data in self.features[dataset]:
                                
                '''
                    Extraction of charges of each molecule, 
                    just pointing the column number

                    notation -> ..., columnIndex
                    extracts the whole specific column by specifying 
                    the col. index
                '''
                self.charges[dataset].append(feature_data[..., self.features_map["partialcharge"]])
            
            '''
                flatten the array of each molec. char vector to one vector
            '''
            # print(len(self.features['validation']))
            try:
                self.charges[dataset] = np.concatenate([charge.flatten() for charge in self.charges[dataset]])
            except:
                raise ValueError(f"Empty array was provided for charges in the dataset {dataset}")
            
        self.charges_std = {'training':self.charges['training'].std(), 'validation':self.charges['validation'].std(), 'testing':self.charges['testing'].std()}

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
            bi = b * self.config.batch_size
            bj = (b + 1)*self.config.batch_size
            if b == self.num_batches[set_name] - 1:
                bj = self.dset_sizes[set_name]
            yield bi, bj

    
    
    def g_batch(self, dataset = 'training', indices = range(0,10), rotation = None):
        x = []
        for index in indices:
            '''
                coordinates and features of one 
                complex
            '''

            coords_index = self.coords[dataset][index] if rotation is None else np.dot(self.coords[dataset][index], rotation)
            
            feature_index = self.features[dataset][index]
            x.append(self.to_box(coords_index, feature_index))
            
        x = np.vstack(x)
        x[..., self.features_map['partialcharge']] /= self.charges_std[dataset]
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
        ''' Representing the coordinates as 3d grid with 21 Angstons diameter
            for one molecular complex
        '''

        coords_num = len(coords)
        f_shape = features.shape
        features_num = f_shape[1]

        box_size = ceil(2*max_dist + 1)
        box_width = box_size / 2

        # '''
        #     change coordinates of atoms to be relative to the center of the box
        # '''

        grid_coords = (coords) / grid_resolution 


        # copies the arrays and casts its elements as integer
        grid_coords = grid_coords.round().astype(int)

        # detect which atoms are in the box 
        # axis = 1 indicates the values in the rows would be checked
        # since the shape of the coords - (N, 3)
        inside_box = ((grid_coords > -box_width) & (grid_coords < box_width)).all(axis = 1)

        grid = np.zeros((1, box_size, box_size, box_size, features_num), dtype = np.float32)
        
        for (x, y, z), f in zip(grid_coords[inside_box], features[inside_box]):
            grid[0, x, y, z] += f

        return grid
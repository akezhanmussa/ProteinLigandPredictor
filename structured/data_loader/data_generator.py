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

# to make numpy be able to raise the error if in case of invalid error
np.seterr(invalid = 'raise')


global_index = 0

logger = logging.getLogger(name = "whole_process_log")

class DataGenerator:
    """The class representation of the data generation process. It is responsible for data formatting and data loading of protein-ligand complexes
    
    :param config: A configuration of the project
    :type config: JSON 
    """
    
    def __init__(self, config):
        self.config = config
        
        # splitting the data by folders
        self.split_data(self.config.data_name, affinity_for_testing= self.config.affinity_for_testing)
        
        # save the rotaion matrices
        self.fill_rotations()
        
        # filling the data of lists from hdf file
        self.fill_data(affinity_for_testing= self.config.affinity_for_testing)
    
        
    def split_data(self, file_name, affinity_for_testing = False):
        """Splits the whole data to different data sets
            
        :param file_name: The name of the data set
        :type file_name: str
        :param affinity_for_testing: whether to include the affinity data for testing folder
        :type affinity_for_testing: bool, optional
        """        
        test_name = self.config.specific_test_name
        
        # path to benchmark core set, but could be any other test cases
        hdf_path = self.config.data_path
        hdf_core_file = f"{hdf_path}/{test_name}"
        hdf_file = hdf_path + '/' + file_name
        path_training = hdf_path + '/' +  "training.hdf"
        path_validation = hdf_path + '/' + "validation.hdf"
        path_testing = hdf_path + '/' + "testing.hdf"
        
        # benchmark set
        if self.config.specific_test_name == "core_set":
            core_set = Options.read_data(os.path.abspath("data_loader/core_pdbbind2013.ids"))
        else:
            core_set = []
            
        # to be sure not to add complexes to testing.hdf twice
        excluded_complexes = defaultdict(lambda: False)
        
        # since testing folder may vary, training and validation files can be untouched 
        if not self.config.testing_mode:
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
                                        
                            self.define_test_case_hdf(hdf_test_file = hdf_core_file, 
                                                    hdf_test_writer = k, 
                                                    affinity_for_testing = affinity_for_testing,
                                                    excluded_complexes = excluded_complexes)
                                                                
                logger.info("The splitting is done")
            else:
                logger.info("The splitting was done before")
        else:
            logger.info("The spplitting will be done only for the test case")
            
            with h5py.File(path_testing, 'w') as k:
                self.define_test_case_hdf(hdf_test_file = hdf_core_file, 
                                            excluded_complexes = excluded_complexes,
                                            hdf_test_writer = k,
                                            affinity_for_testing = affinity_for_testing)
                
            logger.info("The splitting is done")

    @staticmethod
    def define_test_case_hdf(hdf_test_file, hdf_test_writer, affinity_for_testing, excluded_complexes):
        """Defines the particular test case
        
        :param hdf_test_file: the path to the considered test case file 
        :type hdf_test_file: str
        :param affinity_for_testing: True if the affinity has to be extracted, False otherwise (usually, if the model has to be tested for generating affinity, the experimental affinity is not important for testing)
        :type affinity_for_testing: bool
        :param excluded_complexes: the complexes that were excluded 
        :type excluded_complexes: dict, defaultdict as optional
        """
        
        
        with h5py.File(hdf_test_file, 'r') as m:
            for pdb_id in m.keys():
                if excluded_complexes[pdb_id] == False:
                    ds = hdf_test_writer.create_dataset(pdb_id, data = m[pdb_id])
                                        
                if affinity_for_testing:
                    ds.attrs['affinity'] = m[pdb_id].attrs["affinity"]
    
    def fill_rotations(self):
        """Predefines the rotation matrices for data augmentation of the complexes"""
        
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
        """Gets a custom Rotational matrix
        
        :param angle: angle of rotation, should be in range from [-1, 1] 
        :type angle: float
        :param axis: axis of rotation, vector with dimension of three
        :type axis: numpy.ndarray
        :return rotation_matrix: the custom rotational matrix
        :rtype rotation_matrix: numpy.ndarray
        """
        
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
    def rot_random(dataset, indices, rotation_number = 6):
        """Returns a list of random rotations 
        
        :param dataset: type of dataset
        :type dataset: str
        :param indices: the indices of rotational matrices
        :type indices: list
        :param rotation_number: the number of rotations in the pair
        :type rotation_number: int
        :return pair: the list of rotational indices
        :rtype: list
        """
        
        pair = []
        
        if dataset == 'validation' or dataset == 'testing':
            return [0]
        elif dataset == 'training':
            while True:
                # Amid 20 random indices, append to pair till it will have six unique rotaions
                x = np.random.choice(indices, 20)
                # Zero index rotation indicates identity rotation
                x = np.append(x, 0)
                unique_three = set(x)
            
                if len(unique_three) >= rotation_number:
                    break
            
            for index, elem in enumerate(unique_three):
                if index == rotation_number:
                    break
                pair.append(elem)
            
        if len(pair) == 0:
            logger.info("The wrong dataset name, check the name")
        
        return pair
    
    def fill_data(self, affinity_for_testing = True):
        """Fills the dictionaries of attributes (affinity, coords, features, charges) from preprocessed .hdf file
        
        :param affinity_for_testing: True if the affinity has to be extracted, False otherwise (usually, if the model has to be tested for generating affinity, the experimental affinity is not important for testing)
        :type affinity_for_testing: bool
        """

        self.id_s = {}
        self.affinity = {}
        self.coords = {}
        self.features = {}
        self.charges = defaultdict(str)
        self.charges_std = defaultdict(str)
        
        logger.info("The filling data has started")
        
        for dictionary in [self.id_s, self.affinity, self.coords, self.features,self.charges]:
            for dataset in self.config.datasets:
                dictionary[dataset] = []
        
        features_set = self.get_features_names()
        self.features_map = {name:i for i,name in enumerate(features_set)}
        
        # Shrinks the datasets if test case is only considered
        if self.config.testing_mode:
            self.config.datasets = ['testing']
        
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
                        self.id_s[dataset].append(pdb_id)
                        
                        if not affinity_for_testing and dataset == 'testing':
                            continue
                        else:
                            self.affinity[dataset].append(one_set.attrs['affinity'])

            self.id_s[dataset] = np.array(self.id_s[dataset])

            if not affinity_for_testing and dataset == 'testing':
                pass
                #skip
            else:
                self.affinity[dataset] = np.reshape(self.affinity[dataset], (-1, 1))        

            for feature_data in self.features[dataset]:
                                
                """
                    Extraction of charges of each molecule, 
                    just pointing the column number

                    notation -> ..., columnIndex
                    extracts the whole specific column by specifying 
                    the col. index
                """
                          
                self.charges[dataset].append(feature_data[..., self.features_map["partialcharge"]])

            
            # flatten the array of each molec. char vector to one vector
            
            try:
                self.charges[dataset] = np.concatenate([charge.flatten() for charge in self.charges[dataset]])
            except:
                raise ValueError(f"Empty array was provided for charges in the dataset {dataset}")
            
            self.charges_std[dataset] = self.charges[dataset].std()
            
        self.dset_sizes = {dataset: len(self.id_s[dataset]) for dataset in self.config.datasets}
        self.num_batches = {dataset: (size//self.config.batch_size) for dataset, size in self.dset_sizes.items()}

        logger.info("the filling is done")
    
    def batches(self, set_name):
        """Yields the indices for batches
        
        :param set_name: name for dataset
        :type: str
        :return: the start and end of the batch
        :rtype: int, int
        """
        
        for b in range(self.num_batches[set_name]):
            bi = b * self.config.batch_size
            bj = (b + 1)*self.config.batch_size
            if b == self.num_batches[set_name] - 1:
                bj = self.dset_sizes[set_name]
            yield bi, bj

    
    def g_batch(self, dataset = 'training', indices = range(0,10), rotation = None):
        '''Converts the specific range of dataset's complexes to 3d grids
        
        :param dataset: type of dataset
        :type dataset: str
        :param indices: the batch range
        :type indices: list
        :param rotation: the rotation matrix for changing the coordinates of complexes
        :type rotation: numpy.ndarray
        :return x: the list of 3d grid complexes
        :rtype x: list
        '''
        
        
        x = []
        for index in indices:
            # coordinates and features of one complex
            coords_index = self.coords[dataset][index] if rotation is None else np.dot(self.coords[dataset][index], rotation)
            
            feature_index = self.features[dataset][index]
            x.append(self.to_box(coords_index, feature_index))
            
        x = np.vstack(x)
        x[..., self.features_map['partialcharge']] /= self.charges_std[dataset]
        return x 
    
    @staticmethod
    def get_features_names():
        '''Gets the names of features and the order how they were preserved before
        
        :return features: the order of features in the data set
        :rtype features: list
        '''
        atom_classes = Constants.atom_classes.value
 
        features = []
        
        # added the atom names
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
            
        :param coords: the coordinates of the complex
        :type coords: numpy.ndarray
        :param features: the features of the complex
        :type features: list
        :param grid_resolution: resolution of the grid
        :type grid_resolution: float
        :param max_dist: the half of the grid width
        :type max_dist: float, optional
        :return grid: the 3d representation of the complex
        :rtype: numpy.ndarray
        '''

        coords_num = len(coords)
        f_shape = features.shape
        features_num = f_shape[1]

        box_size = ceil(2*max_dist + 1)
        box_width = box_size / 2

        
        # change coordinates of atoms to be relative to the center of the box
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
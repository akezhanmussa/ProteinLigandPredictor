import unittest
import sys
import os
sys.path.append(os.path.abspath("../")) 
from data_loader.data_generator import DataGenerator
from utils.config import get_config
from unittest.mock import patch
import h5py


class TestValidSize(unittest.TestCase):
    '''Test class for checking that
    core set was excluded during the 
    preprocessing and hdf paths exists
    '''
    def setUp(self):
        config, _ = get_config(os.path.abspath('../configs/pln_config.json')) 
        self.config = config
        hdf_path = self.config.data_path
        paths = ["training", "validation","testing","general_set", "390"]
        self.data_path = {}
        
        for path in paths:
            self.data_path[path] = f"{hdf_path}/{path}.hdf"
            
    def test_size_hdf(self):
        
        data_numbers = []
        
        for key in self.data_path.keys():
            with h5py.File(self.data_path[key], 'r') as g:
                data_numbers.append(len(g.keys()))

        # Assert that the overall size of training and validation is less that the size of the general set
        # self.assertLess((data_numbers[0] + data_numbers[1]),data_numbers[3])
        # self.assertGreater((data_numbers[0] + data_numbers[1]),10000)

        self.assertEqual((data_numbers[0] + data_numbers[1]),data_numbers[3])

        # Since the testing can contain some elements from the core set list, size can differemt
        # threshold difference is 10, assert that approximately they have the same size
        # self.assertLess(data_numbers[2], data_numbers[4] + 10)
        self.assertEqual(data_numbers[2], data_numbers[4])
    

    
    def test_file_existance(self):
        for key in self.data_path.keys():
            self.assertEqual(os.path.isfile(self.data_path[key]), True)


if __name__ == "__main__":
    unittest.main()

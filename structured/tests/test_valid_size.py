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
        paths = ["training", "validation","testing","general_set"]
        self.data_path = {}
        
        for path in paths:
            self.data_path[path] = f"{hdf_path}/{path}.hdf"
            
    def test_size_hdf(self):
        
        data_numbers = []
        
        for key in self.data_path.keys():
            with h5py.File(self.data_path[key], 'r') as g:
                data_numbers.append(len(g.keys()))

        # Compare the sum of all three sets size and the general_set size
        self.assertEqual((data_numbers[2]),195)
                             
    def test_file_existance(self):
        for key in self.data_path.keys():
            self.assertEqual(os.path.isfile(self.data_path[key]), True)


if __name__ == "__main__":
    unittest.main()

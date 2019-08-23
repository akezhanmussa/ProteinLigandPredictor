import unittest
import os 



class TestValidFolderCount(unittest.TestCase):
    """Test class for checking whether the 
    big complex was splitted to ten ones
    """
    
    def setUp(self):
        self.path_to_folder = os.path.abspath("pdbbind_data_2018/ligand_poses_from_docking")
        self.folders = os.listdir(self.path_to_folder)
    
    def test_size(self):        
        for folder in self.folders:
            if not folder.endswith("mol2"):
                path_to_folder_complex = f"{self.path_to_folder}/{folder}"
                self.assertEqual(len(os.listdir(path_to_folder_complex)), 10)
    

if __name__ == "__main__":
    unittest.main()
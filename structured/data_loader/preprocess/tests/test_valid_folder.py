import unittest
import os 


class TestValidFolder(unittest.TestCase):
    """Test class for checking that 
    folders of complexes contain
    both the ligand and protein files
    """
    
    def setUp(self):
        self.path_to_folder = os.path.abspath("pdbbind_data_2018/rec_lig")
        self.folders = os.listdir(self.path_to_folder)


    def test_folder(self):
        
        for id,folder_name in enumerate(self.folders):
            folder_contents = os.listdir(f"{self.path_to_folder}/{folder_name}")
            self.assertEqual(folder_contents.count(f"{folder_name}_protein.pdb"), 1)
            self.assertEqual(folder_contents.count(f"{folder_name}_ligand.pdb"), 1)

    def test_unique_folders(self):
        self.assertEqual(len(self.folders), len(set(self.folders)))
        

if __name__ == "__main__":
    unittest.main()
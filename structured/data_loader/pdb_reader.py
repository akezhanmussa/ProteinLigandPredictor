import  os 
import pandas as pd 
import pybel



class Options():
    """Wrapper for reading .txt files         
    
    :param path: the path of the txt file to wrap 
    :type path: str
    """
    
    def __init__(self, path):
        self.path = self.path
        self.unique_pdb = self.read_data(path)
        
    
    @staticmethod
    def read_data(path):
        """Readin the data from .txt file
        
        :param path: the path to a file of pdb codes
        :type path: str
        :return: the unique set of pdb codes
        :rtype: set of str
        """
        
        unique_pdb = set()
        current_path = os.path.abspath(path)
        
        with open(current_path, 'r') as file_reader:
            for line in file_reader:
                unique_pdb.add(line[:4])
                    
        return unique_pdb



    
    

    
        
    
            
            
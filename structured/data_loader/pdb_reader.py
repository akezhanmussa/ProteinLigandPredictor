import  os 
import pandas as pd 



class Options():
    
    def __init__(self, path, csv_path):
        self.path = self.path
        self.unique_pdb = self.read_data(path)
        
    
    @staticmethod
    def read_data(path):
        
        unique_pdb = set()
        current_path = os.path.abspath(path)
        
        with open(current_path, 'r') as file_reader:
            for line in file_reader:
                unique_pdb.add(line[:4])
                    
        return unique_pdb


    @staticmethod
    def format_csv(path):
        pd.set_option('max_rows', 5)
        data = pd.read_csv(path,  index_col = 0)
        
        
    
            
            
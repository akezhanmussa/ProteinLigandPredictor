import sys
import os
# sys.path.append(os.path.abspath("../"))
# from model.ProtLigNet import ProtLigNet
# from utils.config import get_config
# import seaborn as sns
# import pandas as pd
# import csv 
# import tensorflow as tf 
import pandas as pd
import matplotlib.pyplot as plt

'''Visualizer allows to represent the feature 
importance and errors of mutliple training
'''

class Visualizer():
    
    def __init__(self, errors_dir):
        
        errors = os.listdir(errors_dir)
        errors_data = {}
        
        for error in errors:
            errors_data[error] = pd.read_csv(f"{errors_dir}/{error}", index_col = 0)
        
        fig, axarr = plt.subplots(2,2, figsize = (8,8))
        
        
        
        
        


        # self.graph_path = graph_path
        # self.trained_model_path = trained_model_path
        
        
        
    # def features_init(self):
        
        # w_zeroconv =  self.model.graph.get_tensor_by_name('convolution/conv_layer_0/w:0')

        # with tf.Session(graph = self.model.graph) as session:
        #     self.model.saver.restore(session, self.trained_model_path)
        #     self.w_0 = session.run(w_zeroconv)
            
        # print("HERE ", self.w_0.shape)
        
    def box_plot_features(self):
        pass
         
            
if __name__ == "__main__":
    
    # config, _ = get_config(os.path.abspath('configs/pln_config.json'))
    
    # model = ProtLigNet(config)
    
    # vis = Visualizer(model, os.path.abspath("saved_models/"))
    # vis.features_init()
    
    
    
    
    
    
        
        
        
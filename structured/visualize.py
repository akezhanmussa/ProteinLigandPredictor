import sys
import os
sys.path.append(os.path.abspath("../"))
from model.ProtLigNet import ProtLigNet
from utils.config import get_config
import seaborn 
import pandas as pd
import csv 
import tensorflow as tf 

class Visualizer():
    
    def __init__(self, model, trained_model_path):
        self.model = model
        self.trained_model_path = trained_model_path
        self.features_init
        
    def features_init(self):
        
        w_zeroconv =  self.model.graph.get_tensor_by_name('convolution/conv_layer_0/w:0')
        
        with tf.Session(graph = self.model.graph) as session:
            self.model.saver.restore(session, self.trained_model_path)
            self.w_0 = session.run(w_zeroconv)
            
        print(self.w_0.shape)
        
    def box_plot_features(self):
        pass
         
            
if __name__ == "__main__":
    config, _ = get_config(os.path.abspath('configs/pln_config.json'))
    
    model = ProtLigNet(config)
    
    vis = Visualizer(model, os.path.abspath("../saved_models"))
    
    
    
        
        
        
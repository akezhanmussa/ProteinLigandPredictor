from itertools import combinations, permutations,product
from math import cos,sin,sqrt,pi
import numpy as np
import csv
from datetime import datetime

rotation_matrices = []

def save_to_csv(train_error, val_error, path):
        
        current_time = datetime.now().strftime("%d:%m:%Y_%H:%M")
        
        with open(f"{path}/errors_{current_time}.csv", 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["train_error", "val_error"])
            writer.writeheader()
            
            for (t_err, v_err) in zip(train_error, val_error):
                writer.writerow({'train_error':t_err, 'val_error': v_err})
            

def get_rotation_matrix(axis, angle):
        '''Rotation matrix for any angle 
        around any axis
        
        angle should be in range from [-1, 1] 
        '''
        
        # normalize the units of the rotation axis
        if not np.dot(axis, axis) == 0:
            axis = axis / sqrt(np.dot(axis, axis))
        
        u1, u2, u3 = axis
        
        c = cos(angle)
        s = sin(angle)
        v = 1 - c
        
        rotation_matrix = np.array([[u1*u1*v + c,u1*u2*v - u3*s, u1*u3*v + u2*s],
                                    [u1*u2*v + u3*s, u2*u2*v + c, u2*u3*v - u1*s],
                                    [u1*u3*v - u2*s, u2*u3*v + u1*s, u3*u3*v + c]])
        return rotation_matrix
    

    
def fill_rotations_2():
        '''Predefine the rotation matrices
        for data augmentation of the complexes
        '''
        global rotation_matrices

        # The first rotation matrix is just the identity mapping
        rotation_matrices = [get_rotation_matrix(np.array([1,1,1]), 0)]
            
        # 27 different rotations axices, with 100 angles for each
        # matrix contains 2700 rotations 
        for (x,y,z) in product([0,-1,1], repeat = 3):
            for n in range(1, 100):
                axis = np.array([x,y,z])
                angle = n * (pi/50)
                rotation_matrices.append(get_rotation_matrix(axis, angle))   
                
def rot_random(dataset, indices):
    
    pair = []
    
    if dataset == 'validation':
        return [0]
    elif dataset == 'training':
        while True:
            x = np.random.choice(indices, 20)
            x = np.append(x, 0)
            unique_three = set(x)
        
            if len(unique_three) >= 4:
                break
        
        for index, elem in enumerate(unique_three):
            if index == 4:
                break
            pair.append(elem)
        
    if len(pair) == 0:
        print("The wrong dataset name, check the name")
    
    return pair
          
def fill_rotations():
        '''Predefine the rotation matrices
        for data augmentation of the complexes
        '''
        global rotation_matrices
        # The first rotation matrix is just the identity mapping
        rotation_matrices = [get_rotation_matrix(np.array([1,1,1]), 0)]
        
        # Add the rotations around the primal axices
        # for n*pi/4 angles 
        for x in range(3):
            for n in range(1, 4):
                axis = np.zeros([3])
                angle = n * (pi/4)
                axis[x] = 1
                rotation_matrices.append(get_rotation_matrix(axis, angle))
                
        # Add the rotations around the four space diagonals
        # for n*pi/3 angles
        space_axices = np.array([[1,1,-1],[1,1,1],[1,-1,-1],[1,-1,1]])
        
        for axis in space_axices:
            for n in range(1, 6):
                angle = n*(pi/3)
                rotation_matrices.append(get_rotation_matrix(axis, angle))
                

if __name__ == "__main__":
    save_to_csv([1,2,3,4],[1,2,3,4], "/home/mussaa/Downloads/AkeFiles/Files/project_ver2/new_version/ProteinLigandPredictor/structured/logs/training_errors")
    
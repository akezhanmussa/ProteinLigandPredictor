from itertools import combinations, permutations
from math import cos,sin,sqrt,pi
import numpy as np

rotation_matrices = []

def get_rotation_matrix(axis, angle):
        '''Rotation matrix for any angle 
        around any axis
        
        angle should be in range from [-1, 1] 
        '''
        
        # normalize the units of the rotation axis
        axis = axis / sqrt(np.dot(axis, axis))
        
        u1, u2, u3 = axis
        
        c = cos(angle)
        s = sin(angle)
        v = 1 - c
        
        rotation_matrix = np.array([[u1*u1*v + c,u1*u2*v - u3*s, u1*u3*v + u2*s],
                                    [u1*u2*v + u3*s, u2*u2*v + c, u2*u3*v - u1*s],
                                    [u1*u3*v - u2*s, u2*u3*v + u1*s, u3*u3*v + c]])
        return rotation_matrix
    
def rot_random(indices):
        pairs = []
        for (x,y) in combinations(indices, 2):
            choices = list(permutations((0,x,y)))
            pairs.append(choices[np.random.choice(len(choices), 1)[0]])
        return pairs
    
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
    fill_rotations()
    el = (rot_random(range(1, len(rotation_matrices))))
    index = np.random.choice(len(el), 1)
    print(index)
    print(el[index[0]])
    
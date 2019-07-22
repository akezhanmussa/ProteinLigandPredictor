from enum import Enum 
import os

'''
    Constants needed 
    during the computations
    most of them are important 
    for feature definement
'''


class Constants(Enum):
    NAMED_PROPS = ['hyb', 'heavyvalence', 'heterovalence',
                                'partialcharge']
    atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104))), 'metal')
            ]

    SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
    smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor',
                             'ring']
    
    HDF_PATH =  os.path.join(os.pardir, "chimera/preprocessed_data/")

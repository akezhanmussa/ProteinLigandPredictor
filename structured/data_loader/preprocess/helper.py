import pandas as pd
import os 
import h5py
import pybel
import numpy as np
import csv
import json 
from math import modf
from decimal import Decimal 
from constant import Global_var
import sys
from custom_logger import Logger



metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))

atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]

NUM_ATOM_CLASSES = len(atom_classes)
ATOM_CODES = {}
SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]


NAMED_PROPS = ['hyb', 'heavyvalence', 'heterovalence']
logger = Logger(path = os.path.abspath("logs"), name = 'not_read_by_pybel', warning_to_file = True)

def atom_codes_init():
    global atom_classes
    global ATOM_CODES
    global NUM_ATOM_CLASSES

    for code, (atom, name) in enumerate(atom_classes):
        if type(atom) is list:
            for a in atom:
                ATOM_CODES[a] = code
        else:
            ATOM_CODES[atom] = code

    NUM_ATOM_CLASSES = len(atom_classes)


def encode_num(atomic_num):
    global NUM_ATOM_CLASSES
    global ATOM_CODES
    encoding = np.zeros(NUM_ATOM_CLASSES)
    try:
        encoding[ATOM_CODES[atomic_num]] = 1.0
    except:
        pass
    return encoding

def is_file_empty(path_to_file):    
    return True if os.stat(path_to_file).st_size == 0 else False

def read_json(id, db_path = Global_var.DB_PATH.value):
    path_charge = os.path.abspath(f"{db_path}/{id}/{id}.json")
    data = {}
    with open(path_charge, 'r') as r:
        data = json.load(r)
    return data

def to_string(nums):    
    result = ""
    for num in nums:
        # _, decimal = modf(num)
        str_num = str(num)
        dec_str_num = Decimal(str_num)
        final_str = str(float(round(dec_str_num, 3)))   
        result += final_str     
    return result



def extract_features(miss_container, id, molcode = None, db_path = Global_var.DB_GENERAL_PATH.value, relative_central_mass = np.zeros((3,))):
    '''Extract features from the complex
    
    param molcode: pointing whether it is a protein or ligand
    param id: the name of the complex code 
    param db_path: the path where the complex is located
    param relative_central_mass: the central of the ligand
    
    return coords, features, central_mass: decoded coords, features of the complex, central_mass is estimated only for ligands


'''
    atom_codes_init()
    
    mol_id = id if molcode == -1 else f"{id}_ligand"

    path_to_molecule = os.path.abspath("{}/{}/{}.mol2".format(db_path,id,mol_id))
    
    try:
        assert os.path.isfile(path_to_molecule)
    except:
        logger.info(f"{mol_id} was excluded")
        if molcode == 1:
            return [], [], []
        else:
            return [], [] 
        
    if (is_file_empty(path_to_molecule)): 
        if molcode == 1:
            return [],[],[]
        else:
            return [],[]

    try:
        molecule = next(pybel.readfile('mol2', path_to_molecule))
    except Warning:
        logger.info(f"{mol_id} was excluded")
        return [], []    
        
    coords = []
    features = []
    heavy_atoms = []

    mismatch = 0
    charges_db = read_json(id, db_path = db_path)
    
    total_coord_mismatch = 0
    
    for i, atom in enumerate(molecule):
        if atom.atomicnum > 1:
            atomic_features = [atom.__getattribute__(prop) for prop in NAMED_PROPS]
            charge = sys.maxsize
            
            if (molcode == 1):
                central_mass = central_mass_compute(molecule)
                charge = atom.__getattribute__('partialcharge')
            elif (molcode == -1):
                if not are_coordinates_acceptable(atom.coords, relative_central_mass, exclude_radius=10): 
                    total_coord_mismatch += 1
                    continue
                charge = get_probs(miss_container, id, atom, charges_db)
            
            # if the charge was assigned to max, it means that the 
            # particular charge does not exist in our json file
            if (charge == sys.maxsize):
                mismatch += 1
                continue
            
            
            atomic_features.append(charge)
            heavy_atoms.append(i)
            coords.append(atom.coords)
            
            features.append(
                np.concatenate((
                    encode_num(atom.atomicnum),
                    atomic_features
                ))
            )
            
    
    coords = np.array(coords, dtype = np.float64)
    features = np.array(features, dtype = np.float64)

    try:
        assert features.shape[0] > 0 and coords.shape[0] > 0
        features = np.hstack((features, molcode * np.ones((len(features), 1))))
        features = np.hstack([features, find_smarts(molecule)[heavy_atoms]])
    except:
        coords = np.array([0])
        features = np.array([0])

    if molcode == 1:
        return coords, features, central_mass 
    else:
        return coords, features


def get_probs(miss_container, id, atom, charges_db):
    key = ""
    str_num = to_string(atom.coords)
    key += str_num
    charge = 0
    try:
        assert charges_db[key]
        charge = charges_db[key]
    except:
        if id not in miss_container:
            miss_container[id] = []
        miss_container[id].append(atom.residue.name)
        charge = sys.maxsize   
    
    return charge

'''

    Method_1: how to read affinity data 

'''
def prepare_affinity_data():
    path_to_affinity_db = os.path.abspath("pdbbind_data_2018/refined-set/index/INDEX_refined_data.2018")
    path_to_affinity_csv = os.path.abspath("chimera/charges/affinity.csv")
    csv_rows = [["pdbid", "-logKd/Ki"]]
    
    with open(path_to_affinity_db, 'r') as f:
        rows = f.readlines()
        index = 0
        with open(path_to_affinity_csv, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames = ["index", "pdbid", "-logKd/Ki"])
            writer.writeheader()

            for row in rows:
                splitted_row = row.split()
                if (splitted_row[0] == "#"):
                    continue
                else:
                    writer.writerow({"index":index, "pdbid":splitted_row[0], "-logKd/Ki":splitted_row[3]})
                    index += 1


def pattern():
    global SMARTS
    PATTERNS = []
    for smarts in SMARTS:
        el = pybel.Smarts(smarts)
        PATTERNS.append(el)
    return PATTERNS


def find_smarts(molecule):
        # path = os.path.abspath("chimera/{}.mol2".format(id))
        # molecule = next(pybel.readfile('mol2', path))

        """Find atoms that match SMARTS patterns.

        Parameters
        ----------
        molecule: pybel.Molecule

        Returns
        -------
        features: np.ndarray
            NxM binary array, where N is the number of atoms in the `molecule`
            and M is the number of patterns. `features[i, j]` == 1.0 if i'th
            atom has j'th property
        """

        __PATTERNS = compile_smarts()

        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given'
                            % type(molecule))

        features = np.zeros((len(molecule.atoms), len(__PATTERNS)))

        for (pattern_id, pattern) in enumerate(__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))),
                                       dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

def compile_smarts():
        SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]


        __PATTERNS = []

        for smarts in SMARTS:
            __PATTERNS.append(pybel.Smarts(smarts))
        return __PATTERNS 




def are_coordinates_acceptable(atom_coords, relative_central_mass, exclude_radius = 9):
    """Checking whether the specific atom in 10 A distance relative 
    to the ligand central mass
    
    param atom_coords: the considered coordinates of the atom 
    param relative_central_mass: the ligand central mass
    return: the boolean result 
    """    

    atom_to_np = np.asarray(atom_coords)
    coord_difference = atom_to_np - relative_central_mass
    atom_to_np = np.abs(coord_difference)
    
    return False if (atom_to_np > exclude_radius).any() else True
    
def central_mass_compute(molecule):
    """Reads the molecule of mol2 file and computes the 
    central mass of the complex

    param molecule : the molecule of mol2file
    return: the np.array central_mass of the complex
    """    
    sum = 0
    center = np.array([0,0,0], dtype = np.float64)
        
    for atom in molecule.atoms:
        sum += atom.atomicmass
        center += np.asarray(atom.coords)*atom.atomicmass
    
    return center / sum 

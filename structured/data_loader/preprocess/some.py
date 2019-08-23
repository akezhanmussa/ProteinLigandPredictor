
# # Attempt to add some charge
import chimera
from chimera import openModels
from chimera import runCommand
from Midas import write
import os
import logging
from config import get_config

# from constant import Global_var

PATH_TO_CONFIG = os.path.abspath("data_config.json")
config = get_config(PATH_TO_CONFIG)
DB_PATH = "{}/{}".format(config["data_path"], config["data_name"])

runCommand("addh")
    # runCommand("addcharge nonstd chargeModel 12sb")
try:
    runCommand("addcharge")
    #all chargeModel {}".format(AMBER14SB))
except Exception:
    raise Exception("The adding charge operation was not gone properly, check the structure of the file")
    
m = openModels.list()[0]
name = m.name[0:-12]    
name_file = name + "_protonated_charged.pdb"
path_pdb = os.path.abspath("{}/{}/{}".format(DB_PATH,name,name_file))
path_mol2 = os.path.abspath("{}/{}/{}.mol2".format(DB_PATH,name,name))
write(m, None, path_pdb)    
os.system("babel -ipdb {} -omol2 {}".format(path_pdb,path_mol2))

# if __name__ == "__main__":
    


# pybel.write(m, filename = )

# import chimera
# m = chimera.openModels.list()[0]
# residues = m.residues
# for residue in residues:
#     print([atom.name for atom in residue.atoms])
#     print("-----------")
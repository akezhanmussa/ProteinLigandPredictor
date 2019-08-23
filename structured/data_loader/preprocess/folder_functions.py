import os 
from constant import Global_var


class FolderFunctions:
    """Class provides the functions 
    for making manipulations with folders containing 
    protein-ligand complexes
    """
    
    @staticmethod
    def merge_ligand_complexes_with_proteins(name = "390", path_to_proteins = os.path.abspath(f'{Global_var.DB.value}/39_protein_pdb'), path_to_ligands = os.path.abspath('pdbbind_data_2018/ligand_poses_from_docking')):
        """Merging ligand complexes with proteins by 
        putting them on the same folders
        
        param name: the final folder with proteins and ligands
        param path_to_proteins: the path to proteins
        param path_to_ligands: the path to ligands
        """
        merged_folder = os.path.abspath(f"{Global_var.DB.value}/{name}")
        
        if not os.path.isdir(merged_folder):
            os.mkdir(merged_folder)
            
        protein_files = sorted(os.listdir(path_to_proteins))
        ligand_files = sorted(os.listdir(path_to_ligands))
        
        # index for iteration both protein and ligand folders simultaneously
        union_index = 0
        
        try:
            assert len(protein_files) == len(ligand_files)
        except Exception:
            raise AssertionError("Check the equality of protein and ligand folders")

        while union_index < len(protein_files):            
            protein_complex = protein_files[union_index]
            base_code = protein_complex[:4]
            path_to_protein = f"{path_to_proteins}/{protein_complex}"
            path_to_ligand = f"{path_to_ligands}/{ligand_files[union_index]}"
            
            for ligand in os.listdir(path_to_ligand):            
                if not ligand == f"{base_code}.mol2":
                    # path to folder which contains both protein and ligand
                    lig_prot_folder = os.path.abspath(f"{merged_folder}/{ligand[:-5]}")
                    if not os.path.isdir(lig_prot_folder):
                        os.mkdir(lig_prot_folder)
                    shutil.copy(path_to_protein, lig_prot_folder)
                    shutil.copy(f"{path_to_ligand}/{ligand}", lig_prot_folder)
                                    
                    os.rename(f"{lig_prot_folder}/{protein_complex}", f"{lig_prot_folder}/{ligand[:-5]}_protein.pdb")
                    os.rename(f"{lig_prot_folder}/{ligand}", f"{lig_prot_folder}/{ligand[:-5]}_ligand.mol2")

            union_index += 1                        
        
        print("DONE WITH MERGING PROTEIN AND LIGAND FOLDERS")
    
    @staticmethod
    def split_many_ligands(path_to_folder = os.path.abspath(f'{Global_var.DB.value}/ligand_poses_from_docking')):
        """Split the files containing several ligands to different files
        
        param path_to_folder: the directory of multi-ligand complexes    
        """
        index = 0
        
        for complex_file in os.listdir(path_to_folder):
            
            complex_path = f"{path_to_folder}/{complex_file}"
            complex_folder = f"{path_to_folder}/{complex_file[0:4]}"
            complex_index = 0
            can_read_till_bond = False
            reading_after_bond = False
            data = ""
            
            if not os.path.isdir(complex_folder):
                os.mkdir(complex_folder)
            
            os.system(f"babel {complex_path} {complex_folder}/{complex_file[0:4]}_.mol2 -m")
            os.rename(complex_path, f"{complex_folder}/{complex_file}")
            
        print("DONE WITH SPLITTING")
            
    @staticmethod   
    def convert_folders_to_mol2(path_to_folder = os.path.abspath("pdbbind_data_2018/rec_lig"), specific_ending = "_ligand.pdb"):
        """Convert the folder of specific complexes to 
        mol2 format 
        
        param path_to_folder: the directory of specific complexes
        param specific_ending: the commond ending of specific complexes 
        """
        
        all_folders = os.listdir(path_to_folder)
        
        index = 0
        converted_number = 0
        already_converted = 0
        
        while (index < len(all_folders)):
            for folder_file in os.listdir(f"{path_to_folder}/{all_folders[index]}"):
                if folder_file.endswith(specific_ending):
                    new_dir_folder_file = f"{path_to_folder}/{all_folders[index]}/{folder_file[0:4]}_ligand.mol2"
                    if (os.path.isfile(new_dir_folder_file)):
                        already_converted += 1
                        continue
                    dir_folder_file = f"{path_to_folder}/{all_folders[index]}/{folder_file}"
                    os.system(f"babel -ipdb {dir_folder_file} -omol2 {new_dir_folder_file}")
                    converted_number += 1

            index += 1

        print("DONE WITH CONVERSION")
        print(f"TOTAL NUMBER OF CONVERSIONS IS {converted_number}")
        print(f"TOTAL NUMBER OF FILES WHICH WERE CONVERTED BEFORE IS {already_converted}")



    @staticmethod
    def divide_by_folders(path_to_folder = os.path.abspath("pdbbind_data_2018/rec_lig")):
        """Divide complexes to each specific folder
        in case if they are spreaded in the same directory
        
        
        param path_to_folder: the directory of the folder to divide
        
        """
        
        all_folders = os.listdir(path_to_folder)
        
        index = 0
        
        while (index < len(all_folders)):
            ligand_file = ""
            protein_file = ""
            
            
            base_name = all_folders[index][0:4].lower()
            
            if not os.path.isdir(f"{path_to_folder}/{base_name}"):
                os.mkdir(f"{path_to_folder}/{base_name}")
            
            if all_folders[index].endswith("rec.pdb"):
                os.rename(f"{path_to_folder}/{all_folders[index]}",f"{path_to_folder}/{base_name}/{base_name}_protein.pdb")
            elif all_folders[index].endswith("lig.pdb"):
                os.rename(f"{path_to_folder}/{all_folders[index]}", f"{path_to_folder}/{base_name}/{base_name}_ligand.pdb")
            
            index += 1
            
        print("DONE WITH RENAMING")
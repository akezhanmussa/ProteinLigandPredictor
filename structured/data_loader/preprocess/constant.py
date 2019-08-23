from enum import Enum 
import os 

class Global_var(Enum):

    DB_PATH = os.path.abspath("pdbbind_data_2018/refined-set")
    DB_GENERAL_PATH = os.path.abspath("pdbbind_data_2018/general-set")
    DB_CORE_PATH = os.path.abspath("pdbbind_data_2018/coreset")
    DB_REC_LIG_PATH = os.path.abspath("pdbbind_data_2018/rec_lig")
    DB = os.path.abspath("pdbbind_data_2018")
    METRIC_CONVERT = {
        'mM':10**-3,
        'uM':10**-6,
        'pM':10**-12,
        'nM':10**-9,
        'fM':10**-15,
        'M':10**0
    }


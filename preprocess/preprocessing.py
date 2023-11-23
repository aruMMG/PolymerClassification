import os
import sys

# Get the directory of the current script (process.py)
current_dir = os.path.dirname(os.path.realpath(__file__))

# Add the parent directory (A) to the Python path
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

# =========================================
# to change dpt file to csv file


# from utils import dpt2csv
# dpt_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/dpt/Others/COC"
# dpt2csv(dpt_path=dpt_path)
# dpt_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/dpt/Others/COP"
# dpt2csv(dpt_path=dpt_path)
# dpt_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/dpt/Others/PA"
# dpt2csv(dpt_path=dpt_path)
# dpt_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/dpt/Others/PLA"
# dpt2csv(dpt_path=dpt_path)
# ============================================
# Make intensity zero for all data before 672 weve length. Singapore data has intensity values for 400 to 672, whereas data collected in warwick does not have


# from utils import zero672
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/PMMA/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PMMA"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/Others/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/Others"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/PET/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PET"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/PP/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PP"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/PC/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PC"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/PVC/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PVC"
# zero672(csv_path=csv_path, save_dir=save_path)
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv/ABS/"
# save_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/ABS"
# zero672(csv_path=csv_path, save_dir=save_path)
# ==============================================
# Create a single npy file from all csv files. This interpolate takes 400-4000 original format and interpolate it to each intiger wavelength.


# from utils import createNPY
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_warwick/PP_new_18removed/"
# createNPY(csv_path=csv_path, save_name="PP_new_18remove.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PP/"
# createNPY(csv_path=csv_path, save_name="PP.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/Others/"
# createNPY(csv_path=csv_path, save_name="Others.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PET/"
# createNPY(csv_path=csv_path, save_name="PET.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PS/"
# createNPY(csv_path=csv_path, save_name="PS.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PVC/"
# createNPY(csv_path=csv_path, save_name="PVC.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/ABS/"
# createNPY(csv_path=csv_path, save_name="ABS.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PMMA/"
# createNPY(csv_path=csv_path, save_name="PMMA.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PC/"
# createNPY(csv_path=csv_path, save_name="PC.npy")

# ==============================================
# Create a single npy file from all csv files. This interpolate takes 400-4000 original format and interpolate it to each intiger wavelength.

# from utils import createNPY_raw
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/X_PE/"
# createNPY_raw(csv_path=csv_path, save_name="X_PE_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PP/"
# createNPY_raw(csv_path=csv_path, save_name="PP_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/others/"
# createNPY_raw(csv_path=csv_path, save_name="others_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/ABS/"
# createNPY_raw(csv_path=csv_path, save_name="ABS_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PMMA/"
# createNPY_raw(csv_path=csv_path, save_name="PMMA_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PS/"
# createNPY_raw(csv_path=csv_path, save_name="PS_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PVC/"
# createNPY_raw(csv_path=csv_path, save_name="PVC_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PET/"
# createNPY_raw(csv_path=csv_path, save_name="PET_raw.npy")
# csv_path = "/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/FTIR/csv/PC/"
# createNPY_raw(csv_path=csv_path, save_name="PC_raw.npy")

# ===============================================================
# From npy of 1867 to 4000
# from utils import to4000



# ===================================================
# Apply baseline correction


# from utils import asls_dataset
# import numpy as np
# array = np.load("data/data_warwick/HDPE.npy")
# asls_dataset(array, "data/data_warwick/HDPE_baseline.npy")
# array = np.load("data/data_warwick/LDPE.npy")
# asls_dataset(array, "data/data_warwick/LDPE_baseline.npy")
# array = np.load("data/data_warwick/PET.npy")
# asls_dataset(array, "data/data_warwick/PET_baseline.npy")
# array = np.load("data/data_warwick/PP.npy")
# asls_dataset(array, "data/data_warwick/PP_baseline.npy")

# ==========================================================
# Combine npy file.

import numpy as np
arr4 = np.load("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_warwick/PET.npy")
arr6 = np.load("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_singapore/csv_672/PET.npy")
arr = np.vstack((arr4, arr6))
print(arr.shape)
np.save("/home/singh_a_WMGDS.WMG.WARWICK.AC.UK/wmg/polymer_sort/classification/data/data_war_sngp/W_S2_PET.npy", arr)
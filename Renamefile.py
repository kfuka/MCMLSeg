import glob
import os

mask_one_dir = "/mnt/Create_multi_mask/mask_one/"
output_dir = "/mnt/Create_multi_mask/mask_one_renamed/"

file_list = glob.glob(mask_one_dir + "*")
# print(len(file_list))
for patient_file in file_list:
    patient_num, slice_num = patient_file.split("/")[-1].split("_")
    # print(patient_num, slice_num)
    if int(patient_num) >= 46:
        os.rename(patient_file, output_dir + str(int(patient_num) + 1).zfill(3) + slice_num)
    else:
        os.rename(patient_file, output_dir + str(patient_num).zfill(3) + "_" + slice_num)

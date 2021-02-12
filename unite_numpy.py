import csv
import glob

import numba as nb
import numpy as np
from joblib import Parallel, delayed

img_folder = "/mnt/Create_multi_mask/img/"
mask_folder = "/mnt/Create_multi_mask/mask/"
img_one = "/mnt/Create_multi_mask/img_one/"
mask_one = "/mnt/Create_multi_mask/mask_one/"

structure_number = 34
structure_hierarchy_file = "hierarchy.csv"
# skip_str = [6, 30, 31]
structure_color = (np.linspace(1, structure_number + 1, structure_number + 1)).astype(np.uint8)


# print(structure_color)


def get_filelist():
    """
    ファイルのリストを得る。特に000.npyをみる。
    :return: ct画像のリスト、マスク画像のリスト、マスクのnumpy arrayを返す。
    """
    mask_pt = glob.glob(mask_folder + "*_000.npy")
    return mask_pt


@nb.jit(nopython=True)
def sum_array(a_mask, hierarchy):
    new_mask = np.zeros((a_mask.shape[1], a_mask.shape[2]), np.float64)
    for i in range(new_mask.shape[0]):
        for j in range(new_mask.shape[1]):
            candidate = a_mask[:, i, j]
            if np.all(candidate == 0):
                new_mask[i, j] = 0
            else:
                for k in range(len(hierarchy)):
                    num = k + 1
                    c = candidate[list(hierarchy).index(num)]
                    if c != 0:
                        new_mask[i, j] = len(hierarchy) - k
                        break
    return new_mask


def unite_mask(mask_pt, hierarchy):
    def loop_for_a_patient(patient_file):
        patient_num, slice_num_file = patient_file.split("/")[-1].split("_")
        # img_patient = glob.glob(img_folder + pt[patient_num].split("/")[-1].split("_")[0] + "*")
        mask_patient = glob.glob(mask_folder + patient_num + "*")
        for a_slice in mask_patient:
            print("Patient: ", patient_num, "Slice: ", a_slice)
            slice_in_num = a_slice.split("/")[-1].split("_")[1].split(".")[0]
            a_mask = np.load(a_slice)
            new_mask = sum_array(a_mask, hierarchy)

            # new_mask = np.zeros([a_mask.shape[1], a_mask.shape[2]])
            #
            # for i in range(new_mask.shape[0]):
            #     for j in range(new_mask.shape[1]):
            #         candidate = a_mask[:, i, j]
            #         if np.all(candidate == 0):
            #             new_mask[i, j] = 0
            #         else:
            #             for k in range(len(hierarchy)):
            #                 num = k + 1
            #                 c = candidate[list(hierarchy).index(num)]
            #                 if c != 0:
            #                     new_mask[i, j] = len(hierarchy) - k
            #                     break
            np.save(mask_one + str(patient_num).zfill(3) + "_" + slice_in_num + ".npy", new_mask)

    Parallel(n_jobs=-1)(delayed(loop_for_a_patient)(patient_file) for patient_file in mask_pt)
    # for patient_file in mask_pt:
    # loop_for_a_patient(patient_file)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(111)
    # ax1.imshow(a_img, cmap="Greys_r")
    # ax1.imshow(new_mask*255/np.max(new_mask), cmap="hsv", vmin=0, vmax=255, alpha=0.2)
    # plt.show()
    return


def main():
    with open(structure_hierarchy_file) as f:
        reader = csv.reader(f)
        hierarchy = np.array([int(row[0]) for row in reader])
    mask_pt = get_filelist()
    unite_mask(mask_pt, hierarchy)


if __name__ == '__main__':
    main()

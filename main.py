""" BIPED dataset augmentation processes

This script has the whole augmentation methods described in the paper entitle
Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection *WACV2020*
which can be download in: https://arxiv.org/pdf/1909.01955.pdf
"""
import os
import json
import numpy as np
import cv2 as cv

from data_augmentation import augment_data

def cv_imshow(img,title='image'):
    print(img.shape)
    cv.imshow(title,img)
    cv.waitKey(0)
    cv.destroyAllWindows()

def list_data(base_dirs=None,data_name="BIPED", simple_list=False):
    # list BIPED augmented data in a .lst file

    # base_img_dir, base_gt_dir = base_dirs
    dataset_name = data_name
    img_base_dir = 'edges/imgs/train/rgbr/aug'
    gt_base_dir = 'edges/edge_maps/train/rgbr/aug'
    save_file = os.path.join(base_dirs, dataset_name)
    files_idcs = []

    print(f"Peparing the list of {data_name}...")
    if simple_list:
        # img_dirs = os.path.join(save_file, img_base_dir)
        # gt_base_dir = os.path.join(save_file, gt_base_dir)
        for full_path in os.listdir(os.path.join(save_file, img_base_dir)):
            file_name = os.path.splitext(full_path)[0]
            files_idcs.append(
                (os.path.join(img_base_dir + '/' + file_name + '.png'),
                 os.path.join(gt_base_dir + '/' + file_name + '.png'),))
    # For BIPED dataset
    else:
        for dir_name in os.listdir(os.path.join(save_file, img_base_dir)):
            # img_dirs = img_base_dir + '/' + dir_name
            img_dirs = img_base_dir + '/' + dir_name
            for full_path in os.listdir(os.path.join(save_file, img_dirs)):
                file_name = os.path.splitext(full_path)[0]
                files_idcs.append(
                    (os.path.join(img_dirs + '/' + file_name + '.jpg'),
                     os.path.join(gt_base_dir + '/' + dir_name + '/' + file_name + '.png'),))
    # save files

    save_path = os.path.join(save_file, 'train_pair.lst')
    with open(save_path, 'w') as txtfile:
        json.dump(files_idcs, txtfile)

    print("Saved in> ", save_path)

    # Check the files

    with open(save_path) as f:
        recov_data = json.load(f)
    idx = np.random.choice(200, 1)
    tmp_files = recov_data[15]
    img = cv.imread(os.path.join(save_file, tmp_files[0]))
    gt = cv.imread(os.path.join(save_file, tmp_files[1]))
    cv_imshow(img, 'rgb image')
    cv_imshow(gt, 'gt image')

def main(dataset_dir):

    # Data augmentation
    augment_both = True  # to augment the RGB and target (edge_map) image at the same time
    augment_data(base_dir=dataset_dir, augment_both=augment_both, use_all_type=True)

    # Data augmentation list maker>>> train_pair.lst
    list_data(dataset_dir,"BIPED")



if __name__=='__main__':

    # Once the BIPED datset is downloaded, put the localization of the dataset
    # for example if the data is in /home/user_name/datasets/BIPED
    #  put "/home/user_name/datasets"
    BIPED_main_dir ="/home/user_name/datasets"

    main(dataset_dir=BIPED_main_dir)
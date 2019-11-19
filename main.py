""" BIPED dataset augmentation processes

This script has the whole augmentation methods described in the paper entitle
Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection *WACV2020*
which can be download in: https://arxiv.org/pdf/1909.01955.pdf
"""
from data_augmentation import augment_data

def main(dataset_dir):
    augment_both = True  # to augment the RGB and target (edge_map) image at the same time
    augment_data(base_dir=dataset_dir, augment_both=augment_both, use_all_type=True)


if __name__=='__main__':
     # Once the BIPED datset is downloaded, put the localization of the dataset
     # for example if the data is in /home/user_name/datasets/BIPED
    #  put /home/home/user_name
    base_dir = '/home/home/user_name'
    main(dataset_dir=base_dir)
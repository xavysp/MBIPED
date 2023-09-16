# BIPED
Barcelona Images for Perceptual Edge Detection (BIPED) Dataset---descriptions.

```diff 
    We are in the second version of BIPED, termed as BIPEDv2 (This is the ultimate version of BIPED).
    We strongly suggest this last version.
```

![useful image](/figs/BIPED_banner.png)

## Dataset Generation

It contains 250 outdoor images of 1280$\times$720 pixels each. These images have been carefully annotated by experts on the computer vision field, hence no redundancy has been considered. In spite of that, all results have been cross-checked several times in order to correct possible mistakes or wrong edges by just one subject. This dataset is publicly available as a benchmark for evaluating edge detection algorithms. The generation of this dataset is motivated by the lack of edge detection datasets, actually, there is just one dataset publicly available for the edge detection task published in 2016 (MDBD: Multicue Dataset for Boundary Detection---the subset for edge detection). The level of details of the edge level annotations in the BIPED's images can be appreciated looking at the GT, see Figs above. 

BIPED dataset has 250 images in high definition. Thoses images are already split up for training and testing. 200 for training and 50 for testing.


[Download BIPED dataset here](https://drive.google.com/drive/folders/1lZuvJxL4dvhVGgiITmZsjUJPBBrFI_bM?usp=sharing)

# BIPED Data Augmentation

Once the dataset, [BIPEDv2](https://www.kaggle.com/datasets/xavysp/biped?select=BIPEDv2) is downloaded and unzipped, 
use main.py to augment BIPED images as suggest in [DexiNed paper](https://arxiv.org/pdf/1909.01955.pdf),
when the augmentation process is done your file with the list of augmentation
is also created, train_pair.lst, so you just make the following
changes then.
```
# For example if BIPED dir is in /home/user_name/datasets/BIPED
# put "/home/user_name/datasets"
BIPED_main_dir = "/home/user_name/datasets"

    python main.py

```


# License

This Dataset is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree to our license terms. However, if any of our images are infringing any privacy or rights, help free to contact us and we will remove immediately.

If you need more information, [Dont hesitate and contact me :)](https://xavysp.github.io)

# Citation
Please cite our dataset if you find helpful,
```
@InProceedings{soria2020dexined,
    title={Dense Extreme Inception Network: Towards a Robust CNN Model for Edge Detection},
    author={Xavier Soria and Edgar Riba and Angel Sappa},
    booktitle={The IEEE Winter Conference on Applications of Computer Vision (WACV '20)},
    year={2020}
}
```

The extended version and BIPEDv2:
```
@article{soria2023dexined_ext,
title = {Dense extreme inception network for edge detection},
journal = {Pattern Recognition},
volume = {139},
pages = {109461},
year = {2023},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2023.109461},
url = {https://www.sciencedirect.com/science/article/pii/S0031320323001619},
author = {Xavier Soria and Angel Sappa and Patricio Humanante and Arash Akbarinia},
keywords = {Edge detection, Deep learning, CNN, Contour detection, Boundary detection, Segmentation}
}
```

# Acknowledgement




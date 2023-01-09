# MSCDA

[MSCDA: Multi-level Semantic-guided Contrast Improves Unsupervised Domain Adaptation for Breast MRI Segmentation in Small Datasets]

## Architecture

### Overview

![MSCDA](https://github.com/ShengKuangCN/MSCDA/blob/main/figures/architecture.png)

### Multi-level Semantic-Guided Contrast

<img src="https://github.com/ShengKuangCN/MSCDA/blob/main/figures/multi_level_semantic_guided_contrast.png?raw=true" width="550" height="500" />

## Prerequisites

  pip install -r requirements.txt

## Usages

### Train & Testing
We offer several training/testing options as below
* For scenario (--scenario): 
   * '1': Scenario 1 (T2W-T1W)
   * '2': Scenario 2 (T1W-T2W)
* For tasks (--task): 
   * '4': number of source domain subjects 4
   * '8': number of source domain subjects 8
   * '11': number of source domain subjects 11
* For batchsize (--batchsize, default 32)
* For training/testing epoch (--epoch, default 100)
* For GPU allocation (--gpuid, e.g., '1,2')


#### example
For MSCDA training:
    
    python train\train_MSCDA.py --scenario 1 --task 4 --batchsize 16 --epoch 200 --gpu 1

For testing model after applying MSCDA:

    python test\test_MSCDA.py --scenario 1  --task 4 --batchsize 16 --epoch 200 --gpu 1

### Dataset
The datasets are not open access due to the current data-sharing protocal. If you want to run MSCDA based on your own datasets, you can either 
 
(1) reorganize your datasets:
  Step 1. Resample each image and the corresponding mask to 256*256 and save them in the order of [image, mask] as a NumPy file (.npz).
  Step 2. Organize files into folders './data/dataset_1' and './data/dataset_2'. Files should be lised as follows:
    
    +-- dataset_1/2
    |   +-- DYN/VISTA
    |   |   +-- Subject_001
    |   |   |   +-- 1.npz
    |   |   |   +-- 2.npz
    |   |   |   +-- ...
    |   |   |
    |   |   +-- Subject_002
    |   |   |   +-- 1.npz
    |   |   |   +-- 2.npz
    |   |   |   +-- ...
    
 or
 
 (2) use the core file './uda/MSCDA.py' to fit your own domain adaptation project.
 
## Results

### Method performance

Method | Scenario | Task | DSC(%) | JSC(%) | PRC(%) | SEN(%)
:---: | :---: | :---: | :---: | :---: | :---: | :---:
Src-Only  | 1 | S11 | 71.9 | 58.4 | 83.1 | 69.2
Src-Only  | 1 | S8 | 69.1 | 56.1 | 90.9 | 61.8
Src-Only  | 1 | S4 | 54.9 | 41.3 | 94.1 | 44.3
Src-Only  | 2 | S11 | 70.0 | 58.0 | 90.5 | 63.7
Src-Only  | 2 | S8 | 74.3 | 65.4 | 88.5 | 73.4
Src-Only  | 2 | S4 | 70.3 | 57.2 | 95.7 | 60.0
MSCDA  | 1 | S11 | 88.6 | 79.9 | 86.5 | 92.3
MSCDA  | 1 | S8 | 89.2 | 81.0 | 89.3 | 89.9
MSCDA  | 1 | S4 | 87.2 | 78.0 | 92.4 | 83.6
MSCDA  | 2 | S11 | 83.1 | 71.8 | 88.7 | 79.5
MSCDA  | 2 | S8 | 84.0 | 73.2 | 91.7 | 78.8
MSCDA  | 2 | S4 | 83.4 | 72.5 | 98.0 | 73.8
Supervised | 1 | - | 95.8 | 92.8 | 98.0 | 94.7 
Supervised | 2 | - | 96.0 | 93.0 | 96.2 | 96.5

### Segmentation comparison

<img src="https://github.com/ShengKuangCN/MSCDA/blob/main/figures/seg.png?raw=true" width="550" height="500" />

### 

## Citation

    @misc{https://doi.org/10.48550/arxiv.2301.02554,
      doi = {10.48550/ARXIV.2301.02554},
      url = {https://arxiv.org/abs/2301.02554},
      author = {Kuang, Sheng and Woodruff, Henry C. and Granzier, Renee and van Nijnatten, Thiemo J. A. and Lobbes, Marc B. I. and Smidt, Marjolein L. and Lambin, Philippe and Mehrkanoon, Siamak},
      keywords = {Quantitative Methods (q-bio.QM), Machine Learning (cs.LG), FOS: Biological sciences, FOS: Biological sciences, FOS: Computer and information sciences, FOS: Computer and information sciences, I.2; I.5},
      title = {MSCDA: Multi-level Semantic-guided Contrast Improves Unsupervised Domain Adaptation for Breast MRI Segmentation in Small Datasets},
      publisher = {arXiv},
      year = {2023},
    }


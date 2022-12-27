from yacs.config import CfgNode as CN
import os
import numpy as np

cwd = os.getcwd()
sep = os.path.sep

cfg = CN()

# * ------------------------------------------------------------------------------------------ *
# * -- DATA settings
# * ------------------------------------------------------------------------------------------ *

cfg.DATA = CN()
cfg.DATA.DATASET_1 = CN()
cfg.DATA.DATASET_1.PATH = sep.join([cwd, 'data', 'dataset_1', 'image']) + sep
cfg.DATA.DATASET_1.VAL_PATH = cfg.DATA.DATASET_1.PATH
cfg.DATA.DATASET_1.NUM_SUBJECTS = 11
cfg.DATA.DATASET_1.PROTOCOL = 'DYN'

cfg.DATA.DATASET_2 = CN()
cfg.DATA.DATASET_2.PATH = sep.join([cwd, 'data', 'dataset_2', 'image']) + sep
cfg.DATA.DATASET_2.VAL_PATH = cfg.DATA.DATASET_2.PATH
cfg.DATA.DATASET_2.NUM_SUBJECTS = 142
cfg.DATA.DATASET_2.PROTOCOL = 'VISTA'
PROTOCOL_PATTERN = {'DYN': '^dyn_HR', 'VISTA': '^VISTA'}
cfg.DATA.DATASET_2.PROTOCOL_PATTERN = PROTOCOL_PATTERN[cfg.DATA.DATASET_2.PROTOCOL]

cfg.DATA.LOADER = CN()
cfg.DATA.LOADER.NUM_WORKERS = 4
cfg.DATA.LOADER.PREFETCH_FACTOR = 2
cfg.DATA.LOADER.DROP_LAST = True
cfg.DATA.LOADER.PIN_MEMORY = True


# cross-validation settings
d1_sub = {
    'TRT_T1W_n4': [0, 1, 2, 3],
    'TRT_T1W_n8': [0, 1, 2, 3, 4, 5, 6, 7],
    'TRT_T1W_n11': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'TRT_T2W_n4': [0, 1, 2, 3],
    'TRT_T2W_n8': [0, 1, 2, 3, 4, 5, 6, 7],
    'TRT_T2W_n11': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

d2_sub = {
    'D2_T1W_n90_f1': np.arange(1, 94).tolist(),
    'D2_T1W_n90_f2': np.arange(1, 48).tolist() + np.arange(94, 142).tolist(),
    'D2_T1W_n90_f3': np.arange(48, 142).tolist(),
    'D2_T2W_n90_f1': np.arange(1, 94).tolist(),
    'D2_T2W_n90_f2': np.arange(1, 48).tolist() + np.arange(94, 142).tolist(),
    'D2_T2W_n90_f3': np.arange(48, 142).tolist()
}
d2_sub_test = {
    'D2_T1W_n90_f1': np.arange(94, 142).tolist(),
    'D2_T1W_n90_f2': np.arange(48, 94).tolist(),
    'D2_T1W_n90_f3': np.arange(1, 48).tolist(),
    'D2_T2W_n90_f1': np.arange(94, 142).tolist(),
    'D2_T2W_n90_f2': np.arange(48, 94).tolist(),
    'D2_T2W_n90_f3': np.arange(1, 48).tolist()
}


def get_config():
    """Get a yacs CfgNode object with default values."""
    cfg_c = cfg.clone()
    return cfg_c

from yacs.config import CfgNode as CN
from configs import config


cfg = config.get_config()

# * ------------------------------------------------------------------------------------------ *
# * -- MODEL settings: MSCDA
# * ------------------------------------------------------------------------------------------ *

cfg.gpu_ids = [0]
cfg.ckpt_dir = 'E:\\Sheng\\2022_Breast_UDA\\ckpt\\BreastUDA\\'

# Training data settings
cfg.dataset = [cfg.DATA.DATASET_1.PATH, cfg.DATA.DATASET_2.PATH]
cfg.modality = ['VISTA', 'DYN']
cfg.task = 11
cfg.fold = 1
cfg.dataset_name = ['TRT_T1W_n{}'.format(cfg.task), 'D2_T2W_n90_f{}'.format(cfg.fold)]

cfg.subject_id = [config.d1_sub[cfg.dataset_name[0]], config.d2_sub[cfg.dataset_name[1]]]
cfg.subject_id_test = [[], config.d2_sub_test[cfg.dataset_name[1]]]

cfg.frame = [None, [0]]
cfg.aug = ['weak', 'strong']
cfg.supervised_aug = False
cfg.val_during_training_flag = True
cfg.freeze_bn = False

# Training setting
cfg.target_retrain = False
cfg.queue_source = True
cfg.queue_target = False
cfg.no_lsgan = False
cfg.dice_weight = None
cfg.lr = 0.01
cfg.beta1 = 0.5
cfg.lr_policy = 'lambda'
cfg.load_epoch = 0
cfg.epoch_count = 1
cfg.niter = 100
cfg.niter_decay = 0
cfg.lr_decay_iters = 0
cfg.batch_size = 30 * len(cfg.gpu_ids)
cfg.save_epoch_interval = 20

cfg.pixel_queue_size = 32768
cfg.centroid_queue_size = 4096
cfg.pixel_n_samples_per_image_per_class = 8
cfg.num_classes = 2
cfg.n_anchors = 16  # anchors per image per class
cfg.n_negative_pixels = 4096  # negative samples per anchor
cfg.n_negative_centroids = 1024  # negative samples per anchor
cfg.n_negative_c2c = 1024
cfg.temperature = 0.07  # 0.07
cfg.label_threshold = 0.95

cfg.consistency_rampup = 30
cfg.ema_decay = 0.999

cfg.lambda_cons = 1
cfg.lambda_p2p_s = 0
cfg.lambda_p2p_t = 2
cfg.lambda_p2c_s = 0
cfg.lambda_p2c_t = 2
cfg.lambda_c2c_s = 0
cfg.lambda_c2c_t = 2


# Deeplab
cfg.net = CN()
cfg.net.pretrain = 'E:\\Sheng\\2022_Breast_UDA\\ckpt\\SourceOnly\\{}\\netS_A_deeplab_contrast_iter40.pth'.format(cfg.dataset_name[0])
cfg.net.name = 'deeplab_contrast'
cfg.net.backbone = 'resnet50'
cfg.net.input_nc = 1
cfg.net.output_nc = (16, 32, 64, 128)
cfg.net.embed_nc = 128
cfg.net.num_classes = 2
cfg.net.dropout = 0
cfg.net.freeze_bn = False
cfg.net.sync_bn = False
cfg.net.init_type = 'kaiming'

# prediction head
cfg.net_pred = CN()
cfg.net_pred.pretrain = False
cfg.net_pred.name = 'head'
cfg.net_pred.input_nc = 128
cfg.net_pred.output_nc = 128
cfg.net_pred.init_type = 'kaiming'
cfg.net.embed_nc = 128



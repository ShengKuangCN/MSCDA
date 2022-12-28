import re
from datetime import datetime
import os
import numpy as np
import torch
from torch.utils.data import RandomSampler
from torchvision import transforms
import argparse
import sys
sys.path.append(os.getcwd())

from data.dataset import MRImageData
from uda.MSCDA import MSCDA


def test_MSCDA(cfg):
    # Data loader
    transform = transforms.Compose([transforms.ToTensor()])
    target = MRImageData(
        folder=cfg.dataset[1], is_supervised=True, modality=cfg.modality[1], transform=transform,
        subject_id=cfg.subject_id_test[1], aug=False, frame=cfg.frame[1], req_path=True)
    print('Target: {}'.format(len(target)))
    loader = torch.utils.data.DataLoader(
        target, batch_size=cfg.batch_size, num_workers=4, persistent_workers=True, shuffle=False,
        collate_fn=torch.utils.data.dataloader.default_collate, pin_memory=False, prefetch_factor=2, drop_last=False
    )

    # Initialize model
    model = MSCDA(cfg)
    model.load_networks(epoch=cfg.load_epoch)
    model._eval()
    dice_list = []
    ja_list = []
    hd_list = []
    pr_list = []
    sn_list = []
    roi_dice_list = []
    pixel_list = []
    path_list = []
    subject_list = []
    index_list = []
    class_score_list = []
    epoch_start_time = datetime.now()
    for i, data in enumerate(loader):
        model.set_input(data, test=True)
        dice, ja, hd, pr, sn, flg_idx, pixel_idx, class_score = model.get_dice_eval(is_test=True)
        roi_dice = [dice[mi] for mi, v in enumerate(flg_idx) if v == 1]
        dice_list += dice
        ja_list += ja
        hd_list += hd
        pr_list += pr
        sn_list += sn
        path_list += data[2]
        roi_dice_list += roi_dice
        pixel_list += pixel_idx
        class_score_list += class_score

    print(
        '[{}] Test Iter-{}  Time Taken: {}, All-Dice: {:.5}, Dice: {:.5}, Ja: {:.5}, HD: {:.5}, Pr: {:.5}, Sn: {:.5}'.format(
            datetime.now(), i, datetime.now() - epoch_start_time, np.average(dice_list),
            np.average(roi_dice_list),
            np.sum(np.array(ja_list) * np.array(pixel_list)) / np.sum(pixel_list),
            np.average(np.array(hd_list).astype(np.float32)[~np.isnan(np.array(hd_list).astype(np.float32))]),
            np.sum(np.array(pr_list) * np.array(pixel_list)) / np.sum(pixel_list),
            np.sum(np.array(sn_list) * np.array(pixel_list)) / np.sum(pixel_list),
        ))

    for fn in path_list:
        index_list.append(int(fn.split('\\')[-1].split('_')[0].replace('.npz', '')))
        subject_list.append(int(re.search('Subject_(\\d+)', fn).group(1)))

    np.savez(model.save_path[:-4]+'.npz', dice=dice_list, ja=ja_list, hd=hd_list, pr=pr_list, sn=sn_list, pixel=pixel_list, index=index_list, subject=subject_list)
    print('Evaluation data saved !')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--scenario', help="scenario: '1' or '2'", default=1)
    parser.add_argument('-t', '--task', help="tasks: '4/8/11'", default=4)
    parser.add_argument('-f', '--fold', help="cross-validation fold: '1/2/3'", default=1)
    parser.add_argument('-b', "--batchsize", help="batch size", default=32)
    parser.add_argument("-e", "--epoch", help="load epoch", default=100)
    parser.add_argument("-g", "--gpuid", help="run model on gpu id, e.g., '1,2'")

    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpuid
    if args.scenario == 1:
        from configs.Scenario1_config import cfg
    elif args.scenario == 2:
        from configs.Scenario2_config import cfg
    else:
        raise 'Undefined scenario.'
    cfg.task = args.task
    cfg.fold = args.fold
    cfg.gpu_ids = np.arange(0, len(args.gpuid.split(','))).tolist()
    cfg.batch_size = args.batchsize * len(cfg.gpu_ids)
    cfg.batch_size_val = cfg.batch_size
    cfg.load_epoch = args.epoch

    test_MSCDA(cfg)

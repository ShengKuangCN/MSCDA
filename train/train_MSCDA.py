from datetime import datetime
import os
import numpy as np
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import argparse
import sys
sys.path.append(os.getcwd())

from uda.MSCDA import MSCDA
from data.dataloader import CombinedImageLoader


def train_MSCDA(cfg):
    # Data loader
    transform = transforms.Compose([transforms.ToTensor()])
    print(cfg)

    # Training loader
    loader = CombinedImageLoader(
        dataset=cfg.dataset, modality=cfg.modality, transform=transform, batch_size=cfg.batch_size, num_workers=4,
        subject_id=[cfg.subject_id[0], cfg.subject_id[1]], aug=cfg.aug, frame=cfg.frame)

    # Initialize MSCDA model
    model = MSCDA(cfg)
    if not os.path.exists(model.data_dir):
        os.mkdir(model.data_dir)
    logger = SummaryWriter(log_dir=model.data_dir)

    if cfg.load_epoch > 1:
        model.load_networks(epoch=cfg.load_epoch)
        print('Load checkpoint:', cfg.load_epoch)
        model.epoch = cfg.load_epoch + 1
        print('Model epoch:', model.epoch)
    model._train()
    num_steps = 0

    # Training
    for epoch in range(cfg.epoch_count, cfg.niter + cfg.niter_decay + 1):
        epoch_start_time = datetime.now()
        epoch_loss = np.zeros((1, 11))
        epoch_loss_avg = np.zeros((1, 11))
        model._train()
        for i, data in enumerate(loader):
            num_steps += 1
            if not isinstance(cfg.aug, list):
                model.set_input(data)
            else:
                model.set_input_aug(data)
            model.optimize_parameters()

            # record loss and derived parameters
            loss, ptr_idx = model.get_loss()
            if not (None in loss):
                epoch_loss += loss
                epoch_loss_avg = epoch_loss / (i + 1)
                print('[{}] {}/{}-{}  T:{}, L:{:.3}:, S:{:.3}, C:{:.3}, Reg:{:.3}, Ctr:{:.3}, p2p:{:.3}/{:.3}, p2c: {:.3}/{:.3}, c2c: {:.3}/{:.3}, ptr: {}/{}, lr: {}'.format(
                    datetime.now(), epoch, cfg.niter + cfg.niter_decay, i, datetime.now() - epoch_start_time,
                    epoch_loss_avg[0, 8], epoch_loss_avg[0, 0], epoch_loss_avg[0, 1], epoch_loss_avg[0, 10],
                    epoch_loss_avg[0, 9], epoch_loss_avg[0, 2], epoch_loss_avg[0, 3], epoch_loss_avg[0, 4],
                    epoch_loss_avg[0, 5], epoch_loss_avg[0, 6], epoch_loss_avg[0, 7],
                    ptr_idx[0], ptr_idx[1], model.optimizer.param_groups[0]['lr']), end='\r')
                logger.add_scalar("Step/Loss/Supervised", loss[0], global_step=num_steps)
                logger.add_scalar("Step/Loss/Consist", loss[1], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-p2p_s", loss[2], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-p2p_t", loss[3], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-p2c_s", loss[4], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-p2c_t", loss[5], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-c2c_s", loss[6], global_step=num_steps)
                logger.add_scalar("Step/Loss/Contrast-c2c_t", loss[7], global_step=num_steps)
                logger.add_scalar("Step/Loss/Total", loss[8], global_step=num_steps)
                logger.add_scalar("Step/Loss/TotalContrast", loss[9], global_step=num_steps)
                logger.add_scalar("Step/Loss/Reg", loss[10], global_step=num_steps)

        if epoch % cfg.save_epoch_interval == 0:
            model.save_networks()

        print(
            '[{}] {}/{}-{}  T:{}, L:{:.3}:, S:{:.3}, C:{:.3}, Reg:{:.3}, Ctr:{:.3}, p2p:{:.3}/{:.3}, p2c: {:.3}/{:.3}, c2c: {:.3}/{:.3}, ptr: {}/{}, lr: {}'.format(
                datetime.now(), epoch, cfg.niter + cfg.niter_decay, i, datetime.now() - epoch_start_time,
                epoch_loss_avg[0, 8], epoch_loss_avg[0, 0], epoch_loss_avg[0, 1], epoch_loss_avg[0, 10],
                epoch_loss_avg[0, 9], epoch_loss_avg[0, 2], epoch_loss_avg[0, 3], epoch_loss_avg[0, 4],
                epoch_loss_avg[0, 5], epoch_loss_avg[0, 6], epoch_loss_avg[0, 7],
                ptr_idx[0], ptr_idx[1], model.optimizer.param_groups[0]['lr']))
        logger.add_scalar("Epoch/Loss/Supervised", epoch_loss_avg[0, 0], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Consist", epoch_loss_avg[0, 1], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-p2p_s", epoch_loss_avg[0, 2], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-p2p_t", epoch_loss_avg[0, 3], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-p2c_s", epoch_loss_avg[0, 4], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-p2c_t", epoch_loss_avg[0, 5], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-c2c_s", epoch_loss_avg[0, 6], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Contrast-c2c_t", epoch_loss_avg[0, 7], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Total", epoch_loss_avg[0, 8], global_step=epoch)
        logger.add_scalar("Epoch/Loss/TotalContrast", epoch_loss_avg[0, 9], global_step=epoch)
        logger.add_scalar("Epoch/Loss/Reg", epoch_loss_avg[0, 10], global_step=epoch)

        # ----
        model.update_learning_rate()
        model.epoch += 1


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
    train_MSCDA(cfg)



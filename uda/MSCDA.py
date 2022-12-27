import itertools
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import network as nt
from network.loss.loss import DiceLoss, get_eval_res, dice_coefficient, DiceCrossEntropyWithRegLoss
from data.dataset import functional_resized_crop_params
import random
import torchvision.transforms.functional as TF


def del_tensor_0_cloumn(Cs):
    idx = torch.all(Cs[..., :] == 0, axis=1)
    index = []
    for i in range(idx.shape[0]):
        if not idx[i].item():
            index.append(i)
    index = torch.tensor(index).cuda()
    Cs = torch.index_select(Cs, 0, index)
    return Cs


class MSCDA(nn.Module):
    def __init__(self, cfg):
        super(MSCDA, self).__init__()

        self.save_path = None
        self.cfg = cfg
        self.device = torch.device('cuda:{}'.format(cfg.gpu_ids[0])) if cfg.gpu_ids else torch.device('cpu')

        # initialize model
        self.net_student = nt.create_model(cfg.net, cfg.gpu_ids)
        if self.training:
            self.net_teacher = nt.create_model(cfg.net, cfg.gpu_ids)
            self.net_pred_head = nt.create_model(cfg.net_pred, cfg.gpu_ids)  # prediction head student

            self.detach_model(self.net_teacher)  # detach teacher net. The weights will be updated by EMA step.
            self.nets = ['net_student', 'net_teacher', 'net_pred_head']

            # initialize centroid queue and relative variables
            # for local centroid
            self.centroid_queue = torch.randn(cfg.centroid_queue_size, cfg.num_classes, cfg.net.embed_nc).cuda()
            self.centroid_queue = nn.functional.normalize(self.centroid_queue, dim=0)
            self.centroid_queue_ptr = torch.zeros(cfg.num_classes, dtype=torch.long).to(self.device)

            # for global centroid
            self.global_centroid = torch.randn(cfg.num_classes, cfg.net.embed_nc).to(self.device)
            self.global_centroid_cnt = 0

            # initialize pixel queue and relative variables
            self.pixel_queue = torch.randn(cfg.pixel_queue_size, cfg.num_classes, cfg.net.embed_nc).cuda()
            self.pixel_queue = nn.functional.normalize(self.pixel_queue, dim=2)
            self.pixel_queue_ptr = torch.zeros(cfg.num_classes, dtype=torch.long).to(self.device)

            # initialize loss function
            self.criterionDice = DiceLoss(weight=cfg.dice_weight)
            self.criterionCE = nn.CrossEntropyLoss()
            self.criterionCons = nn.MSELoss()

            # initialize optimizers
            self.optimizer = torch.optim.Adam(
                itertools.chain(
                    self.net_student.parameters(),
                    self.net_pred_head.parameters()),
                lr=cfg.lr,
                betas=(cfg.beta1, 0.999)
            )

            self.optimizers = [self.optimizer]
            self.schedulers = [nt.get_scheduler(optimizer, cfg) for optimizer in self.optimizers]

        if isinstance(self.cfg.dataset_name, list):
            self.experiment_name = '_to_'.join(self.cfg.dataset_name)
        else:
            self.experiment_name = self.cfg.dataset_name
        self.data_dir = os.path.join(self.cfg.ckpt_dir, self.experiment_name)

        self.img_S = self.img_S_aug = self.img_T = self.img_T_aug = None
        self.label_S = self.label_T = self.label_S_aug = self.label_T_aug = None
        self.feat_stu_S = self.feat_stu_T = self.feat_tea_T = None
        self.seg_stu_S = self.seg_stu_T = self.seg_tea_T = self.seg_tea_T_soft = None
        self.feat_tea_S = self.seg_tea_S = None
        self.proj_stu_S = self.proj_stu_T = None
        self.mask_weights_tea_T = None
        self.mask_ds_stu_S = self.mask_ds_stu_T = self.mask_ds_tea_T = None
        self.mask_ds_tea_S = None
        self.loss_S = self.v_loss_S = self.loss_C = self.v_loss_C = self.v_loss = self.v_loss_contrast_all = None
        self.v_loss_reg = None
        self.loss_contrast_p2p_s = self.v_loss_contrast_p2p_s = None
        self.loss_contrast_p2p_t = self.v_loss_contrast_p2p_t = None
        self.loss_contrast_p2c_s = self.v_loss_contrast_p2c_s = None
        self.loss_contrast_p2c_t = self.v_loss_contrast_p2c_t = None
        self.loss_contrast_c2c_s = self.v_loss_contrast_c2c_s = None
        self.loss_contrast_c2c_t = self.v_loss_contrast_c2c_t = None

        self.centroid_stu_T = self.centroid_stu_S = self.centroid_tea_T = self.centroid_tea_S = None
        self.ent_stu_T = self.ent_tea_T = None
        self.one_hot_seg_stu_T = self.one_hot_seg_tea_T = None
        self.soft = nn.Softmax(dim=1)
        self.label_threshold = 0.5  # self.cfg.label_threshold
        self.iters = 1
        self.epoch = 1

        self.aug_rotation_angle = 0
        self.aug_crop = False
        self.aug_crop_i = None
        self.aug_crop_j = None
        self.aug_crop_h = None
        self.aug_crop_w = None
        self.aug_flip = False

    def set_input(self, input, test=False):
        """ This function is for one-stream setup. """
        self.img_S = input[0].to(self.device)
        self.img_S_aug = self.img_S
        self.label_S = input[1].to(self.device).long()
        self.label_S_aug = self.label_S
        if not test:
            self.img_T = input[2].to(self.device)
            self.img_T_aug = self.img_T
            self.label_T = input[3].to(self.device).long()
            self.label_T_aug = self.label_T

    def set_input_aug(self, input, test=False):
        """
        This function is for two-stream image setup during the training phase.
        The input consists of (at least) eight component:
        1. source image and masks augmented via pipeline 1 and 2
        2. target image and masks augmented via pipeline 1 and 2
        The pipeline is defined in the config file 'cfg.aug'
        """
        self.img_S = input[0].to(self.device)  # source image weakly augmented
        self.label_S = input[1].to(self.device).long()  # source mask weakly augmented
        if not test:
            self.img_S_aug = input[2].to(self.device)  # source image strongly augmented
            self.label_S_aug = input[3].to(self.device).long()  # source mask strongly augmented
            self.img_T = input[4].to(self.device)  # target image weakly augmented
            self.label_T = input[5].to(self.device).long()  # target mask weakly augmented
            self.img_T_aug = input[6].to(self.device)  # target image strongly augmented
            self.label_T_aug = input[7].to(self.device).long()  # target mask strongly augmented

            self.geometric_transform_seed_update()
            self.img_T = self.geometric_transform(self.img_T)
            self.label_T = self.geometric_transform(self.label_T, interpolation=TF.InterpolationMode.NEAREST)
            self.img_S = self.geometric_transform(self.img_S)
            self.label_S = self.geometric_transform(self.label_S, interpolation=TF.InterpolationMode.NEAREST)

    def forward(self):
        """
        student feature = encoder -> projection mlp -> prediction mlp |
        teacher feature = encoder -> projection mlp                 <-|
        """
        # student net: forward source images
        self.feat_stu_S, self.seg_stu_S = self.net_student(self.img_S_aug)
        self.proj_stu_S = self.net_pred_head(self.feat_stu_S)

        # student net: forward target images
        self.feat_stu_T, self.seg_stu_T = self.net_student(self.img_T_aug)
        self.proj_stu_T = self.net_pred_head(self.feat_stu_T)

        # # teacher net: forward source images
        self.feat_tea_S, self.seg_tea_S = self.net_teacher(self.img_S)
        self.feat_tea_S.detach_()
        self.seg_tea_S.detach_()

        # teacher net: forward target images
        self.feat_tea_T, self.seg_tea_T = self.net_teacher(self.img_T)
        self.feat_tea_T.detach_()
        self.seg_tea_T.detach_()

        self.down_sample_mask()

        self.centroid_stu_T = self.calculate_centroid(self.proj_stu_T, self.mask_ds_stu_T.detach(), no_grad=False)  # [batch, class, embed]
        self.centroid_stu_S = self.calculate_centroid(self.proj_stu_S, self.mask_ds_stu_S.detach(), no_grad=False)  # [batch, class, embed]
        self.centroid_tea_T = self.calculate_centroid(self.feat_tea_T, self.mask_ds_tea_T.detach(), no_grad=True)  # [batch, class, embed]
        self.centroid_tea_S = self.calculate_centroid(self.feat_tea_S, self.mask_ds_tea_S.detach(), no_grad=True)  # [batch, class, embed]

    def backward(self):
        # Supervised loss
        loss_S_dice = self.criterionDice(self.seg_stu_S, self.label_S_aug)
        loss_S_ce = self.criterionCE(self.seg_stu_S, self.label_S_aug)
        self.loss_S = (1 + loss_S_dice + loss_S_ce) * 0.5  # -1 < loss_S_dice < 0

        self.v_loss_S = self.loss_S.item()
        self.loss_C = self.criterionCons(F.softmax(self.geometric_transform(self.seg_stu_T), dim=1), F.softmax(self.seg_tea_T, dim=1).detach())

        self.v_loss_C = self.loss_C.item()
        cons_weight = self.get_current_consistency_weight()

        # Supervised Contrastive loss: pixel-to-pixel, pixel-to-centroid
        # -- step 1: sample anchors from current batch
        n_anchors = self.cfg.n_anchors * self.proj_stu_T.shape[0]
        f_stu_S, l_stu_S, _ = self.sample_anchors_from_feat(self.proj_stu_S, self.mask_ds_stu_S, n_anchors)
        f_stu_T, l_stu_T, _ = self.sample_anchors_from_feat(self.proj_stu_T, self.mask_ds_stu_T, n_anchors)
        # -- step 2: sample positive/negative samples from queue
        class_label = np.arange(self.cfg.num_classes)
        pos_pixel_list = []  # class-wise positive samples
        neg_pixel_list = []  # class-wise negative samples
        pos_centroid_list = []  # class-wise positive samples
        neg_centroid_list = []  # class-wise negative samples
        for c in class_label:
            _pos_pixels = self.sample_from_queue(self.pixel_queue, n_samples=n_anchors, class_index=c, inclusive=True)
            _neg_pixels = self.sample_from_queue(self.pixel_queue, n_samples=self.cfg.n_negative_pixels, class_index=c, inclusive=False)
            _pos_centroids = self.sample_from_queue(self.centroid_queue, n_samples=n_anchors, class_index=c, inclusive=True)
            _neg_centroids = self.sample_from_queue(self.centroid_queue, n_samples=self.cfg.n_negative_centroids, class_index=c, inclusive=False)
            pos_pixel_list.append(_pos_pixels)
            neg_pixel_list.append(_neg_pixels)
            pos_centroid_list.append(_pos_centroids)
            neg_centroid_list.append(_neg_centroids)

        # -- step 3: calculate InfoNCE loss: pixel-to-pixel, pixel-to-centroid
        q_pixel_list = f_stu_T + f_stu_S  # combine target and source batch
        label_list = l_stu_T + l_stu_S
        self.loss_contrast_p2p_s = torch.Tensor([0]).to(self.device)
        self.loss_contrast_p2p_t = torch.Tensor([0]).to(self.device)
        self.loss_contrast_p2c_s = torch.Tensor([0]).to(self.device)
        self.loss_contrast_p2c_t = torch.Tensor([0]).to(self.device)

        t_idx = 0
        for q_pixel, label in zip(q_pixel_list, label_list):
            # define q, k, n
            q_pixel = F.normalize(q_pixel, dim=1)
            k_pixel = F.normalize(pos_pixel_list[label][:q_pixel.shape[0], :], dim=1).detach()
            n_pixel = F.normalize(neg_pixel_list[label], dim=1).detach()
            k_centroid_for_p2c = F.normalize(pos_centroid_list[label][:q_pixel.shape[0], :], dim=1).detach()
            n_centroid = F.normalize(neg_centroid_list[label], dim=1).detach()

            # -- pixel-to-pixel loss
            l_p2p_pos = torch.einsum('nc,nc->n', [q_pixel, k_pixel]).unsqueeze(-1)
            l_p2p_neg = torch.einsum('nc,kc->nk', [q_pixel, n_pixel])
            logits_p2p = torch.cat([l_p2p_pos, l_p2p_neg], dim=1)
            logits_p2p /= self.cfg.temperature
            labels_p2p = torch.zeros(logits_p2p.shape[0], dtype=torch.long).to(self.device)
            if t_idx < len(l_stu_T):
                self.loss_contrast_p2p_t += self.criterionCE(logits_p2p, labels_p2p)
            else:
                self.loss_contrast_p2p_s += self.criterionCE(logits_p2p, labels_p2p)

            # -- pixel to centroid loss
            l_p2c_pos = torch.einsum('nc,nc->n', [q_pixel, k_centroid_for_p2c]).unsqueeze(-1)
            l_p2c_neg = torch.einsum('nc,kc->nk', [q_pixel, n_centroid])
            logits_p2c = torch.cat([l_p2c_pos, l_p2c_neg], dim=1)
            logits_p2c /= self.cfg.temperature
            labels_p2c = torch.zeros(logits_p2c.shape[0], dtype=torch.long).to(self.device)
            if t_idx < len(l_stu_T):
                self.loss_contrast_p2c_t += self.criterionCE(logits_p2c, labels_p2c)
            else:
                self.loss_contrast_p2c_s += self.criterionCE(logits_p2c, labels_p2c)
            t_idx += 1

        # Supervised Contrastive loss: centroid-to-centroid
        # -- step 3: calculate InfoNCE loss: centroid-to-centroid
        fc_stu_S = self.clean_centroids(self.centroid_stu_S)
        fc_stu_T = self.clean_centroids(self.centroid_stu_T)
        q_centroid_list = fc_stu_T + fc_stu_S
        self.loss_contrast_c2c_s = torch.Tensor([0]).to(self.device)
        self.loss_contrast_c2c_t = torch.Tensor([0]).to(self.device)
        t_idx = 0
        for q_centroid in q_centroid_list:
            if q_centroid.shape[0] > 0:
                q_centroid = F.normalize(q_centroid, dim=1)
                # define q, k, n
                label = t_idx % self.cfg.num_classes
                k_centroid_for_c2c = F.normalize(pos_centroid_list[label][:q_centroid.shape[0], :], dim=1).detach()
                n_centroid = F.normalize(neg_centroid_list[label][:self.cfg.n_negative_c2c, :], dim=1).detach()

                # -- centroid to centroid loss
                l_c2c_pos = torch.einsum('nc,nc->n', [q_centroid, k_centroid_for_c2c]).unsqueeze(-1)
                l_c2c_neg = torch.einsum('nc,kc->nk', [q_centroid, n_centroid])
                logits_c2c = torch.cat([l_c2c_pos, l_c2c_neg], dim=1)
                logits_c2c /= self.cfg.temperature
                labels_c2c = torch.zeros(logits_c2c.shape[0], dtype=torch.long).to(self.device)
                if t_idx < self.cfg.num_classes:
                    self.loss_contrast_c2c_t += self.criterionCE(logits_c2c, labels_c2c)
                else:
                    self.loss_contrast_c2c_s += self.criterionCE(logits_c2c, labels_c2c)
            t_idx += 1

        self.loss_contrast_p2p_s = (self.loss_contrast_p2p_s / len(f_stu_S)) if len(f_stu_S) > 0 else self.loss_contrast_p2p_s
        self.loss_contrast_p2p_t = (self.loss_contrast_p2p_t / len(f_stu_T)) if len(f_stu_T) > 0 else self.loss_contrast_p2p_t
        self.loss_contrast_p2c_s = (self.loss_contrast_p2c_s / len(f_stu_S)) if len(f_stu_S) > 0 else self.loss_contrast_p2c_s
        self.loss_contrast_p2c_t = (self.loss_contrast_p2c_t / len(f_stu_T)) if len(f_stu_T) > 0 else self.loss_contrast_p2c_t
        self.loss_contrast_c2c_s = (self.loss_contrast_c2c_s / len(fc_stu_S)) if len(fc_stu_S) > 0 else self.loss_contrast_c2c_s
        self.loss_contrast_c2c_t = (self.loss_contrast_c2c_t / len(fc_stu_T)) if len(fc_stu_T) > 0 else self.loss_contrast_c2c_t
        self.v_loss_contrast_p2p_s = self.loss_contrast_p2p_s.item()
        self.v_loss_contrast_p2p_t = self.loss_contrast_p2p_t.item()
        self.v_loss_contrast_p2c_s = self.loss_contrast_p2c_s.item()
        self.v_loss_contrast_p2c_t = self.loss_contrast_p2c_t.item()
        self.v_loss_contrast_c2c_s = self.loss_contrast_c2c_s.item()
        self.v_loss_contrast_c2c_t = self.loss_contrast_c2c_t.item()

        # backward
        loss_contrast_all = 0.5 * self.cfg.lambda_p2p_s * self.loss_contrast_p2p_s + \
            0.5 * self.cfg.lambda_p2p_t * self.loss_contrast_p2p_t + \
            0.5 * self.cfg.lambda_p2c_s * self.loss_contrast_p2c_s + \
            0.5 * self.cfg.lambda_p2c_t * self.loss_contrast_p2c_t + \
            0.5 * self.cfg.lambda_c2c_s * self.loss_contrast_c2c_s + \
            0.5 * self.cfg.lambda_c2c_t * self.loss_contrast_c2c_t
        loss = self.loss_S + cons_weight * self.loss_C + loss_contrast_all

        self.v_loss_contrast_all = loss_contrast_all.item()
        self.v_loss = loss.item()
        loss.backward()

    def predict(self, x=None):
        if x is not None:
            _, self.seg_stu_S = self.net_student(x)
            return self.seg_stu_S
        else:
            _, self.seg_stu_S = self.net_student(self.img_S)
            return self.seg_stu_S

    def predict_with_feature(self):
        feat_embed, self.seg_stu_S, feat_highdim = self.net_student(self.img_S, req_highdim_feat=True)
        one_hot_label_S_aug = torch.zeros(*self.seg_stu_S.shape, device=self.device)
        one_hot_label_S_aug.scatter_(1, self.label_S_aug.unsqueeze(1).to(self.device), 1)
        self.mask_ds_stu_S = F.interpolate(one_hot_label_S_aug, size=feat_embed.shape[2:], mode='nearest')  # supervised label

        return self.seg_stu_S, feat_highdim, feat_embed, self.mask_ds_stu_S

    def down_sample_mask(self):
        """interpolate segmentation masks to fit the size of feature maps"""
        # source segmentation in student net
        one_hot_label_S_aug = torch.zeros(*self.seg_stu_S.shape, device=self.device)
        one_hot_label_S_aug.scatter_(1, self.label_S_aug.unsqueeze(1).to(self.device), 1)
        self.mask_ds_stu_S = F.interpolate(one_hot_label_S_aug, size=self.proj_stu_T.shape[2:], mode='nearest')  # supervised label
        # target segmentation in student net
        self.one_hot_seg_stu_T = F.softmax(self.seg_stu_T, dim=1).ge(self.cfg.label_threshold).type(dtype=torch.uint8)
        self.mask_ds_stu_T = F.interpolate(self.one_hot_seg_stu_T, size=self.proj_stu_T.shape[2:], mode='nearest')  # down-sampled pseudo label,
        # source segmentation in teacher net
        one_hot_label_S = torch.zeros(*self.seg_tea_S.shape, device=self.device)
        one_hot_label_S.scatter_(1, self.label_S.unsqueeze(1).to(self.device), 1)
        self.mask_ds_tea_S = F.interpolate(one_hot_label_S, size=self.feat_tea_S.shape[2:], mode='nearest')  # down-sampled pseudo label,
        # target segmentation in teacher net
        self.one_hot_seg_tea_T = F.softmax(self.seg_tea_T, dim=1).ge(self.cfg.label_threshold).type(dtype=torch.uint8).detach()
        self.mask_ds_tea_T = F.interpolate(self.one_hot_seg_tea_T, size=self.feat_tea_T.shape[2:], mode='nearest')  # down-sampled pseudo label,

    @staticmethod
    def clean_centroids(centroid):
        """
        :param centroid: [batch, n_classes, n_embed]
        :return: [[batch, n_embed], [batch, n_embed], ..]
        """
        c_l = centroid.split(1, dim=1)
        return [del_tensor_0_cloumn(c.squeeze(1)) for c in c_l]

    @torch.no_grad()
    def update_centroid(self):
        # ema global momentum centroids
        self.ema_global_centroid(self.centroid_tea_S)

        # enqueue and dequeue local centroids
        if self.cfg.queue_source:
            self._dequeue_and_enqueue_local_centroid(queue=self.centroid_queue, queue_ptr=self.centroid_queue_ptr, keys=self.centroid_tea_S)
        if self.cfg.queue_target:
            self._dequeue_and_enqueue_local_centroid(queue=self.centroid_queue, queue_ptr=self.centroid_queue_ptr, keys=self.centroid_tea_T)

    @torch.no_grad()
    def update_pixel(self):
        if self.cfg.queue_source:
            self.sample_pixels_to_queue(
                feat_map=self.feat_tea_S, label=self.mask_ds_tea_S, n_samples=self.cfg.pixel_n_samples_per_image_per_class,
                queue=self.pixel_queue, queue_ptr=self.pixel_queue_ptr)
        if self.cfg.queue_target:
            self.sample_pixels_to_queue(
                feat_map=self.feat_tea_T, label=self.mask_ds_tea_T, n_samples=self.cfg.pixel_n_samples_per_image_per_class,
                queue=self.pixel_queue, queue_ptr=self.pixel_queue_ptr)

    def update_threshold(self):
        if self.epoch >= 10:
            self.label_threshold = self.cfg.label_threshold
        else:
            self.label_threshold = (self.epoch - 1) / 10 * (self.cfg.label_threshold - 0.5) + 0.5

    @torch.no_grad()
    def ema(self):
        alpha = min(1 - 1 / (self.iters + 1), self.cfg.ema_decay)
        for ema_param, param in zip(self.net_teacher.parameters(), self.net_student.parameters()):
            ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

    @torch.no_grad()
    def ema_global_centroid(self, keys):
        alpha = min(1 - 1 / (self.iters + 1), self.cfg.ema_decay)
        self.global_centroid = F.normalize(alpha * self.global_centroid + (1 - alpha) * keys.mean(dim=0),
                                           dim=1).detach()  # [class, embed]
        # print(self.global_centroid[:, :3])
        self.global_centroid_cnt += keys.shape[0]

    @torch.no_grad()
    def _dequeue_and_enqueue_local_centroid(self, queue, queue_ptr, keys):
        batch_size = keys.shape[0]
        # loop for each class and batch index
        for c in range(self.cfg.num_classes):
            for b in range(batch_size):
                ptr = int(queue_ptr[c])
                if not (keys[b, c, :] == 0).all():  # make sure the centroid is not empty
                    queue[ptr, c, :] = keys[b, c, :].detach()
                    ptr = (ptr + 1) % self.cfg.centroid_queue_size
                    queue_ptr[c] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue_pixel(self, queue, queue_ptr, keys, c):
        """
        :param keys: [n_samples, n_dims]
        :param c: class index
        """
        # queues: [n_queue_size, n_classes, n_dims]
        n_batches = keys.shape[0]
        if n_batches:
            ptr = int(queue_ptr[c])

            if self.cfg.pixel_queue_size - ptr >= n_batches:
                queue[ptr:ptr + n_batches, c, :] = F.normalize(keys, dim=1)
            else:
                n_batches = self.cfg.pixel_queue_size - ptr
                queue[ptr:ptr + n_batches, c, :] = F.normalize(keys[:n_batches, :], dim=1)
            ptr = (ptr + n_batches) % self.cfg.pixel_queue_size
            queue_ptr[c] = ptr

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
        self.ema()
        self.update_centroid()
        self.update_pixel()
        self.iters += 1

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def get_current_consistency_weight(self):
        if self.cfg.net.pretrain:
            return 1
        else:
            # Consistency ramp-up from https://arxiv.org/abs/1610.02242
            return self.cfg.lambda_cons * self.sigmoid_rampup(self.epoch, self.cfg.consistency_rampup)

    def get_loss(self):
        v_loss = [
            self.v_loss_S, self.v_loss_C,
            self.v_loss_contrast_p2p_s, self.v_loss_contrast_p2p_t,
            self.v_loss_contrast_p2c_s, self.v_loss_contrast_p2c_t,
            self.v_loss_contrast_c2c_s, self.v_loss_contrast_c2c_t,
            self.v_loss, self.v_loss_contrast_all, self.v_loss_reg
        ]
        ptr = [
            self.pixel_queue_ptr.cpu().detach().numpy(),
            self.centroid_queue_ptr.cpu().detach().numpy(),
        ]
        return np.array(v_loss), ptr

    def get_dice_eval(self, is_test=False):
        self.predict()
        if is_test:
            dice, ja, hd, pr, sn, flg_idx, pixel_idx, class_score = get_eval_res(self.seg_stu_S, self.label_S)
            return dice, ja, hd, pr, sn, flg_idx, pixel_idx, class_score
        else:
            dice = list(dice_coefficient(self.seg_stu_S, self.label_S, reduction='none').detach().cpu().numpy())
            flg_idx = self.label_S.cpu().detach().numpy().max(axis=(1, 2))
            pixel_idx = self.label_S.cpu().detach().numpy().sum(axis=(1, 2)).tolist()
            return dice, flg_idx, pixel_idx

    @staticmethod
    def detach_model(net):
        for param in net.parameters():
            param.detach_()

    @staticmethod
    def freeze_bn(net):
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    @staticmethod
    def sigmoid_rampup(current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    @staticmethod
    def calculate_centroid(feat_map, mask, no_grad=True, weight_map=None):
        """
        down-sampled image: (1, 256, 32, 32) -> (1, 256, 1024) -> (1, 1024, 256)*(matmul)
        down-sampled mask : (1, 2, 32, 32)   -> (1, 2, 1024)   -> (1, 2, 1024)*(1, 1024, 256) -> (1, 2, 256) -> l2-norm -> output
        """
        # class_wise_feat_map_list = class_wise_feat_map.chunk(n_class, dim=1)
        feat_map = F.normalize(feat_map, p=2, dim=1)
        t_feat_map = feat_map.reshape((feat_map.shape[0], feat_map.shape[1], -1)).transpose(2, 1)
        if weight_map is not None:
            mask = weight_map.repeat(1, 2, 1, 1) * mask
        t_mask = mask.reshape((mask.shape[0], mask.shape[1], -1)).float()
        centroid = F.normalize(torch.matmul(t_mask, t_feat_map), dim=2)
        if no_grad:
            return centroid.detach()
        else:
            return centroid

    def sample_anchors_from_feat(self, feat_map, label, n_samples, paired_feat_map=None):
        """
        sample pixels from designated feature map
        :param feat_map: [batch, n_embed, H', W']
        :param label: [batch, n_classes, H', W']
        :param n_samples: sampling size
        :param paired_feat_map: [batch, n_embed, H', W'], corresponding feature map in teacher net
        :return: feat_list, label_list, indices_list
            feat_list(list): [[n_class1_samples, n_embed], [n_class2_samples, n_embed], ...]
            label_list(list): [0, 1, ...]
            indices_list(list): [[n_class1_samples], [n_class2_samples], ...]
        """
        feat_map = rearrange(feat_map, 'b d h w -> b (h w) d')
        label = rearrange(label, 'b c h w -> b (h w) c')
        if paired_feat_map is not None:
            paired_feat_map = rearrange(paired_feat_map, 'b d h w -> b (h w) d')
        feat_list = []
        label_list = []
        indices_list = []
        pos_feat_list = []

        for c in range(self.cfg.num_classes):
            mask_c = label[:, :, c] == 1
            feature_c = feat_map[mask_c, :]
            if feature_c.shape[0] > n_samples:
                indices = torch.randperm(feature_c.shape[0])[:n_samples]
                sampled_feature = feature_c[indices, :]
                feat_list.append(sampled_feature)
                label_list.append(c)
                indices_list.append(indices)
                if paired_feat_map is not None:
                    positive_feature = paired_feat_map[mask_c, :][indices, :]  # paired pixel
                    pos_feat_list.append(positive_feature)

            else:
                sampled_feature = feature_c

                if sampled_feature.shape[0] > 0:
                    feat_list.append(sampled_feature)
                    label_list.append(c)
                    indices_list.append(None)
                    if paired_feat_map is not None:
                        positive_feature = paired_feat_map[mask_c, :]
                        pos_feat_list.append(positive_feature)
        if paired_feat_map is not None:
            return feat_list, label_list, indices_list, pos_feat_list
        else:
            return feat_list, label_list, indices_list

    @torch.no_grad()
    def sample_from_queue(self, queue, indices=None, n_samples=None, class_index=None, inclusive=True):
        """

        :param queue: [queue_size, n_classes, n_embeds]
        :param indices: if True, indices should be [n_samples, n_classes], 'n_samples' and 'class_index' will not be used
        :param n_samples:
        :param class_index:
        :param inclusive:
        :return: [n_samples, n_embeds]
        """
        if indices is not None:
            return queue[indices]
        elif n_samples is not None and class_index is not None:
            if inclusive:
                queue_indices = torch.randperm(queue.shape[0])[:n_samples]
                # queue_indices = torch.randint(0, queue.shape[0], [n_samples]).to(self.device)
                samples = queue[queue_indices, class_index, :]
                return samples
            else:
                all_class = torch.arange(queue.shape[1])
                class_indices = all_class[all_class != class_index]
                queue_indices = torch.randperm(queue.shape[0] * class_indices.shape[0])[:n_samples]
                samples = rearrange(queue[:, class_indices, :], 'q c f -> (q c) f')[queue_indices, :]
                return samples

    @torch.no_grad()
    def sample_pixels_to_queue(self, feat_map, label, n_samples, queue, queue_ptr):
        """
        :param feat_map: [B, F, H, W]
        :param label: [B, H, W]
        :param n_samples: number of pixels sampled from each class in each image
        """
        n_batches = feat_map.shape[0]

        feat_map = rearrange(feat_map, 'b d h w -> b (h w) d')
        label = rearrange(label, 'b c h w -> b (h w) c')
        for c in range(self.cfg.num_classes):
            mask_c = label[:, :, c] == 1
            feature_c = feat_map[mask_c, :]
            if feature_c.shape[0] > n_samples:
                indices = torch.randperm(feature_c.shape[0])[:n_samples * n_batches]
                sampled_feature = feature_c[indices, :]
                self._dequeue_and_enqueue_pixel(queue, queue_ptr, sampled_feature, c)
            elif feature_c.shape[0] == 0:
                pass
            else:
                self._dequeue_and_enqueue_pixel(queue, queue_ptr, feature_c, c)

    def geometric_transform_seed_update(self):
        p = 0.5
        if random.random() < p:
            self.aug_rotation_angle = random.randint(-30, 30)
        else:
            self.aug_rotation_angle = 0
        if random.random() < p:
            self.aug_crop = True
            crop_scale = np.min([random.random() * 0.2 + 0.8, 1.0])
            self.aug_crop_i, self.aug_crop_j, self.aug_crop_h, self.aug_crop_w = functional_resized_crop_params(
                img_size=self.img_S.shape[-1], scale=(crop_scale, crop_scale))
        else:
            self.aug_crop = False
        self.aug_flip = True if random.random() < p else False

    def geometric_transform(self, image, interpolation=None):
        if not interpolation:
            interpolation = TF.InterpolationMode.BILINEAR
        t_image = TF.rotate(image, self.aug_rotation_angle, interpolation=interpolation)
        if self.aug_crop:
            t_image = TF.resized_crop(t_image, self.aug_crop_i, self.aug_crop_j, self.aug_crop_h, self.aug_crop_w, self.img_S.shape[-2:], interpolation=interpolation)
        if self.aug_flip:
            t_image = TF.hflip(t_image)
        return t_image

    def save_networks(self, save_name=None):
        for net_name in self.nets:
            net = self.__getattr__(net_name)
            if save_name:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, save_name)
            else:
                save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, self.epoch)
            # data_dir = os.path.join(self.cfg.ckpt_dir, self.experiment_name)
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            save_path = os.path.join(self.data_dir, save_filename)
            print(save_path)
            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()
            torch.save({'state_dict': state_dict, 'epoch': self.epoch, 'cfg': self.cfg}, save_path)

    def load_networks(self, epoch):
        for net_name in self.nets:
            net = self.__getattr__(net_name)

            save_filename = '{}_{}_iter{}.pth'.format(net_name, net.module.name, epoch)
            # self.data_dir = os.path.join(self.cfg.ckpt_dir, self.experiment_name)
            save_path = os.path.join(self.data_dir, save_filename)
            if not self.save_path:
                self.save_path = save_path
            print(save_path)
            ckpt_data = torch.load(save_path)
            if isinstance(net, torch.nn.DataParallel):
                net.module.load_state_dict(ckpt_data['state_dict'])
            else:
                net.load_state_dict(ckpt_data['state_dict'])

    def _eval(self):
        self.net_student.eval()
        self.eval()

    def _train(self):
        self.net_student.train()
        self.train()
        if self.cfg.freeze_bn:
            self.freeze_bn(self.net_student)
            self.freeze_bn(self.net_pred_head)


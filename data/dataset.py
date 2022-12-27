from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
import albumentations as A
import cv2
import torch
import random
import math
import torchvision.transforms.functional as TF


def create_augmentation_pipeline(aug_type):
    aug_p = None
    if aug_type == 'weak':
        aug_p = A.Compose([
            A.ShiftScaleRotate(scale_limit=0, rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REPLICATE),
            A.RandomResizedCrop(height=256, width=256, scale=(0.8, 1.2), p=0.5)
        ])
    elif aug_type == 'strong':
        aug_p = A.Compose([
            A.GaussNoise(var_limit=(0, 0.1), mean=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=0.5),
            A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.5),
        ])
    elif aug_type == 'no':
        aug_p = A.Compose([])
    elif aug_type:
        aug_p = A.Compose([
            A.ShiftScaleRotate(scale_limit=(-0.3, 0.2), rotate_limit=15, p=0.5, border_mode=cv2.BORDER_REPLICATE),
            A.GaussNoise(var_limit=(0, 0.1), mean=0, p=0.225),
            A.GaussianBlur(sigma_limit=(0.5, 1.5), p=0.3),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.5, 0.5), p=0.225),
        ]) if aug_type else None
    return aug_p


def functional_rotation(img, angle, itp_mode):
    interpolation = transforms.InterpolationMode.BILINEAR
    if itp_mode == 'image':
        interpolation = transforms.InterpolationMode.BILINEAR
    elif itp_mode == 'mask':
        interpolation = transforms.InterpolationMode.NEAREST
    image = TF.rotate(img, angle, interpolation=interpolation)
    return image


def functional_resized_crop_params(img_size, scale=(0.8, 1.2), ratio=(0.75, 1.3333333333333333)):
    height = img_size
    width = img_size
    area = height * width

    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if 0 < w <= width and 0 < h <= height:
            i = torch.randint(0, height - h + 1, size=(1,)).item()
            j = torch.randint(0, width - w + 1, size=(1,)).item()
            return i, j, h, w

    # Fallback to central crop
    in_ratio = float(width) / float(height)
    if in_ratio < min(ratio):
        w = width
        h = int(round(w / min(ratio)))
    elif in_ratio > max(ratio):
        h = height
        w = int(round(h * max(ratio)))
    else:  # whole image
        w = width
        h = height
    i = (height - h) // 2
    j = (width - w) // 2
    return i, j, h, w


class MRImageData(Dataset):
    def __init__(self, folder, is_supervised=True, modality='DYN', transform=None, subject_id=None, aug=None, req_path=False, frame=None, norm=True, **kwargs):
        """
        This is the implementation of my private MRI dataset.
        :param folder: str, root directory of the image dataset
        :param is_supervised: abandon
        :param modality: str, the modality of MRI image, e.g., 'DYN', 'VISTA'
        :param transform: torchvision.transforms, the transformation after augmentations
        :param subject_id: list or numpy array, the range of subject id
        :param aug: str or list, augmentation options, e.g., 'weak', 'strong', 'patch'
        :param req_path: boolean, if True return the path list of images
        :param frame: list, the specific frame of the MRI, None if does not exists
        :param norm: boolean, whether apply z-score normalization to the image
        :param kwargs:
        """
        super(MRImageData, self).__init__()
        # check folder
        print('[Dataset] Folder:', folder)
        print('[Dataset] Modality:', modality)
        print('[Dataset] Subject_id:', subject_id)
        print('[Dataset] Frame:', frame)
        print('[Dataset] Aug:', aug)

        assert os.path.exists(folder) is True, 'Folder does not exist: {}'.format(folder)
        self.folder = folder
        self.is_supervised = is_supervised
        self.req_path = req_path
        self.transform = transform
        self.aug_pipeline_pre = None
        self.aug_pipeline_pre1 = None
        self.aug_pipeline_pre2 = None

        self.aug = aug
        if isinstance(aug, list):
            self.aug_pipeline1 = create_augmentation_pipeline(aug[0])
            self.aug_pipeline2 = create_augmentation_pipeline(aug[1])
            print('[Augmentation Pipeline 1]: ', self.aug_pipeline1)
            print('[Augmentation] Pipeline 2]: ', self.aug_pipeline2)
            if len(aug) == 3:
                self.aug_pipeline_pre = create_augmentation_pipeline(aug[2])
                print('[Augmentation Pipeline pre]: ', self.aug_pipeline_pre)
            elif len(aug) == 4:
                self.aug_pipeline_pre1 = create_augmentation_pipeline(aug[2])
                print('[Augmentation Pipeline pre]: ', self.aug_pipeline_pre1)
                self.aug_pipeline_pre2 = create_augmentation_pipeline(aug[3])
                print('[Augmentation Pipeline pre]: ', self.aug_pipeline_pre2)
        else:
            self.aug_pipeline = create_augmentation_pipeline(aug)
            print('[Augmentation Pipeline]: ', self.aug_pipeline)

        self.file_list = []
        self.modality = modality
        self.subject_id = subject_id
        self.norm = norm

        # generate file list with conditions: subject_id, frame
        iter_dir = os.walk(os.path.join(folder, modality))
        for t_dir, t_sub, t_file in iter_dir:
            if not t_sub:
                t_path = t_dir.split(os.sep)  # separate path
                sub_path_idx = t_path.index(modality) + 1  # residual sub path under modality folder
                t_sub_path_l = t_path[sub_path_idx:]  # residual sub path under modality folder
                t_subject = t_sub_path_l[0]
                sub_path = os.sep.join(t_sub_path_l)

                # frame
                frame_qualify_flag = True if not frame else False
                if frame and len(t_sub_path_l) == 3:
                    frame_str = [str(fr) for fr in frame]
                    if t_sub_path_l[-1] in frame_str:
                        frame_qualify_flag = True

                # subject
                subject_qualify_flag = True if not subject_id else False
                if subject_id:
                    if int(t_subject[-3:]) in subject_id:
                        subject_qualify_flag = True

                # final decision
                if frame_qualify_flag and subject_qualify_flag:
                    self.file_list += [(modality + os.sep + sub_path + os.sep + n) for n in t_file]

    def __getitem__(self, index):
        f = np.load(self.folder + self.file_list[index], allow_pickle=True)
        f_x = f['arr_0'].astype(dtype="float32")
        if self.norm:
            mean = f_x.mean()
            std = f_x.std()
            f_x = (f_x - mean) / std
        f_y = f['arr_1'].astype(dtype="float32")

        # check and do preliminary processing
        if self.aug_pipeline_pre:
            pre = self.aug_pipeline_pre(image=f_x, mask=f_y)
            img = pre['image']
            mask = pre['mask']
        elif self.aug_pipeline_pre1 and self.aug_pipeline_pre2:
            if random.random() > 0.5:
                pre = self.aug_pipeline_pre1(image=f_x, mask=f_y)
                img = pre['image']
                mask = pre['mask']
            else:
                pre = self.aug_pipeline_pre2(image=f_x, mask=f_y)
                img = pre['image']
                mask = pre['mask']
        else:
            img = f_x
            mask = f_y

        # check how many augmentations need to be done
        if isinstance(self.aug, list):
            aug_res = self.aug_pipeline1(image=img, mask=mask)
            img = aug_res['image']
            mask = aug_res['mask']
            aug_res2 = self.aug_pipeline2(image=img, mask=mask)
            img2 = aug_res2['image']
            mask2 = aug_res2['mask']
            img2 = self.transform(img2)
        elif self.aug:
            aug_res = self.aug_pipeline(image=img, mask=mask)
            img = aug_res['image']
            mask = aug_res['mask']
        img = self.transform(img)
        if (not self.req_path) and (not isinstance(self.aug, list)):
            return img, mask
        elif (not self.req_path) and isinstance(self.aug, list):
            return img, mask, img2, mask2
        elif self.req_path and (not isinstance(self.aug, list)):
            return img, mask, self.folder + self.file_list[index]
        else:
            return img, mask, img2, mask2, self.folder + self.file_list[index]

    def __len__(self):
        return len(self.file_list)



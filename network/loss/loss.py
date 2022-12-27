import numpy as np
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
from skimage.measure import label
import math


hausdorff_sd = cv2.createHausdorffDistanceExtractor()


class DiceLoss(nn.Module):
    """
    Implementation of dice loss
    """

    def __init__(self, smooth=1, reduction='mean', ignore_background=False, weight=None, threshold=None):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=1)
        self.ignore_background = ignore_background
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = (weight / weight.sum()).unsqueeze(1)
        else:
            self.weight = None

    def forward(self, prediction, target):
        """

        :param prediction: the one-hot prediction of shape [MiniBatch, Class, *ImageSize]
        :param target: the ground truth index of shape [MiniBatch, *ImageSize], int64(torch.LongTensor) required
        :return: tensor of [Batch, DiceLoss]
        """
        # print(prediction.shape[2:], target.shape[1:])
        assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[
            0], 'Same shape required !'
        if self.threshold:
            prediction = self.softmax(prediction).ge(self.threshold).type(dtype=torch.uint8)
        else:
            prediction = self.softmax(prediction)
        prediction = prediction.contiguous()
        # print(prediction.device)
        target_one_hot = torch.zeros(*prediction.shape, device=prediction.device)  # prediction.data.clone().zero_()
        target_one_hot.scatter_(1, target.unsqueeze(1).to(prediction.device), 1)
        if self.ignore_background:
            prediction = prediction[:, 1:, :, :]
            target_one_hot = target_one_hot[:, 1:, :, :]
        num_dims = len(prediction.shape[2:])
        sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64) + 1)

        numerator = (prediction * target_one_hot).sum(dim=sum_idx)
        denominator = prediction.sum(dim=sum_idx) + target_one_hot.sum(dim=sum_idx)
        if self.weight is None or self.ignore_background:
            loss = - ((2 * numerator + self.smooth) / (denominator + self.smooth)).sum(dim=1) / prediction.shape[1]
        else:
            loss = torch.matmul(- ((2 * numerator + self.smooth) / (denominator + self.smooth)), self.weight.to(prediction.device))
        # print(loss)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class DiceCrossEntropyWithRegLoss(nn.Module):
    """
    Implementation of dice loss
    """

    def __init__(self, smooth=1, reduction='mean', ignore_background=False, weight=None, threshold=None, margin_weight=0):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.threshold = threshold
        self.softmax = nn.Softmax(dim=1)
        self.ignore_background = ignore_background
        self.margin_weight = margin_weight
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = (weight / weight.sum()).unsqueeze(1)
        else:
            self.weight = None
        self.ce = nn.CrossEntropyLoss(weight=self.weight, reduction=self.reduction).cuda()

    def forward(self, prediction, target):
        """

        :param prediction: the one-hot prediction of shape [MiniBatch, Class, *ImageSize]
        :param target: the ground truth index of shape [MiniBatch, *ImageSize], int64(torch.LongTensor) required
        :return: tensor of [Batch, DiceLoss]
        """
        assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[
            0], 'Same shape required !'
        target_one_hot = torch.zeros(*prediction.shape, device=prediction.device)  # prediction.data.clone().zero_()
        target_one_hot.scatter_(1, target.unsqueeze(1).to(prediction.device), 1)
        target_one_hot_copy = target_one_hot.detach()
        prediction = prediction - self.margin_weight * target_one_hot_copy

        ce_loss = self.ce(prediction, target)
        if self.threshold:
            prediction = self.softmax(prediction).ge(self.threshold).type(dtype=torch.uint8)
        else:
            prediction = self.softmax(prediction)
        prediction = prediction.contiguous()

        if self.ignore_background:
            prediction = prediction[:, 1:, :, :]
            target_one_hot = target_one_hot[:, 1:, :, :]
        num_dims = len(prediction.shape[2:])
        sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64) + 1)

        numerator = (prediction * target_one_hot).sum(dim=sum_idx)
        denominator = prediction.sum(dim=sum_idx) + target_one_hot.sum(dim=sum_idx)
        if self.weight is None or self.ignore_background:
            dice_loss = - ((2 * numerator + self.smooth) / (denominator + self.smooth)).sum(dim=1) / prediction.shape[1]
        else:
            dice_loss = torch.matmul(- ((2 * numerator + self.smooth) / (denominator + self.smooth)), self.weight.to(prediction.device))

        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum()
        elif self.reduction == 'none':
            dice_loss = dice_loss

        loss = (1 + dice_loss + ce_loss) / 2
        return loss


class TverskyLoss(nn.Module):
    """
    Implementation of Tversky loss
    """

    def __init__(self, alpha, beta, smooth=1, reduction='mean', pos_index=1):
        super().__init__()
        self.smooth = smooth
        self.reduction = reduction
        self.pos_index = pos_index
        self.alpha = alpha
        self.beta = beta
        self.softmax = nn.Softmax(dim=1)

    def forward(self, prediction, target):
        """

        :param prediction: the one-hot prediction of shape [MiniBatch, Class, *ImageSize]
        :param target: the one-hot ground truth of shape [MiniBatch, *ImageSize], int64(torch.LongTensor) required
        :return: tensor of [Batch, DiceLoss]
        """
        # print(prediction.shape[2:], target.shape[1:])
        assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[
            0], 'Same shape required !'
        prediction = self.softmax(prediction)
        prediction = prediction.contiguous()[:, self.pos_index, :, :]

        num_dims = len(prediction.shape[1:])
        sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

        intersection = (prediction * target).sum(dim=sum_idx)
        false_positive = ((1 - target) * prediction).sum(dim=sum_idx)
        false_negative = ((1 - prediction) * target).sum(dim=sum_idx)
        loss = - (intersection + self.smooth) / (intersection + self.alpha * false_positive + self.beta * false_negative + self.smooth)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        # print(target_tensor.shape)
        return self.loss(input, target_tensor)


def precision(prediction, target, reduction='mean', label_index=1, smooth=1):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1)
    prediction = prediction[:, label_index, :, :]
    num_dims = len(prediction.shape[1:])
    sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

    numerator = (prediction * target).sum(axis=sum_idx)
    denominator = prediction.sum(axis=sum_idx)
    pre = (numerator + smooth) / (denominator + smooth)
    if reduction == 'mean':
        return pre.mean()
    elif reduction == 'sum':
        return pre.sum()
    elif reduction == 'none':
        return pre
    return None


def sensitivity(prediction, target, reduction='mean', label_index=1, smooth=1):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1)
    prediction = prediction[:, label_index, :, :]
    num_dims = len(prediction.shape[1:])
    sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

    numerator = (prediction * target).sum(axis=sum_idx)
    denominator = target.sum(axis=sum_idx)
    sen = (numerator + smooth) / (denominator + smooth)
    if reduction == 'mean':
        return sen.mean()
    elif reduction == 'sum':
        return sen.sum()
    elif reduction == 'none':
        return sen
    return None


def dice_coefficient_old(prediction, target, reduction='mean', label_index=1, smooth=1):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1)
    prediction = prediction[:, label_index, :, :]
    num_dims = len(prediction.shape[1:])
    sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

    numerator = (prediction * target).sum(axis=sum_idx)
    denominator = prediction.sum(axis=sum_idx) + target.sum(axis=sum_idx)

    dice = (2 * numerator + smooth) / (denominator + smooth)

    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    elif reduction == 'none':
        return dice
    return None


def dice_coefficient(prediction, target, reduction='mean', label_index=1, smooth=1, threshold=0.5):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1).ge(threshold)
    prediction = prediction[:, label_index, :, :]
    num_dims = len(prediction.shape[1:])
    sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

    numerator = (prediction * target).sum(axis=sum_idx)
    denominator = prediction.sum(axis=sum_idx) + target.sum(axis=sum_idx)


    dice = (2 * numerator + smooth) / (denominator + smooth)

    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    elif reduction == 'none':
        return dice
    return None


def jaccard_coefficient(prediction, target, reduction='mean', label_index=1, smooth=1):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1)
    prediction = prediction[:, label_index, :, :]
    num_dims = len(prediction.shape[1:])
    sum_idx = tuple(np.linspace(1, num_dims, num_dims, dtype=np.int64))

    intersection = (prediction * target).sum(axis=sum_idx)
    summation = prediction.sum(axis=sum_idx) + target.sum(axis=sum_idx)
    jaccard = (intersection + smooth) / (summation - intersection + smooth)

    if reduction == 'mean':
        return jaccard.mean()
    elif reduction == 'sum':
        return jaccard.sum()
    elif reduction == 'none':
        return jaccard
    return None


def get_contours(img):
    contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return contours[0]
    else:
        return None


def hausdorff_distance(prediction, target, reduction='mean', label_index=1):
    assert prediction.shape[2:] == target.shape[1:] and prediction.shape[0] == target.shape[0], 'Same shape required !'
    prediction = nn.functional.softmax(prediction, dim=1)
    prediction = prediction[:, label_index, :, :]
    dist_l = []
    for i in range(prediction.shape[0]):
        p = get_contours(prediction[i, :, :].cpu().detach().numpy())
        t = get_contours(target[i, :, :].cpu().detach().numpy())
        if reduction == 'mean':
            if p is not None and t is not None:
                hausdorff = hausdorff_sd.computeDistance(p, t)
                dist_l.append(hausdorff)
        elif reduction == 'none':
            if p is not None and t is not None:
                hausdorff = hausdorff_sd.computeDistance(p, t)
                dist_l.append(hausdorff)
            else:
                dist_l.append(None)

    return dist_l


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class LargeMarginInSoftmaxLoss(nn.CrossEntropyLoss):
    r"""
    https://github.com/tk1980/LargeMarginInSoftmax/blob/37bf331c016f6e9e96c66cc4690a62e7d5984f80/models/modules/myloss.py
    This combines the Softmax Cross-Entropy Loss (nn.CrossEntropyLoss) and the large-margin inducing
    regularization proposed in
       T. Kobayashi, "Large-Margin In Softmax Cross-Entropy Loss." In BMVC2019.

    This loss function inherits the parameters from nn.CrossEntropyLoss except for `reg_lambda` and `deg_logit`.
    Args:
         reg_lambda (float, optional): a regularization parameter. (default: 0.3)
         deg_logit (bool, optional): underestimate (degrade) the target logit by -1 or not. (default: False)
                                     If True, it realizes the method that incorporates the modified loss into ours
                                     as described in the above paper (Table 4).
    """

    def __init__(self, reg_lambda=0.3, deg_logit=None,
                 weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super(LargeMarginInSoftmaxLoss, self).__init__(weight=weight, size_average=size_average,
                                                       ignore_index=ignore_index, reduce=reduce, reduction=reduction)
        self.reg_lambda = reg_lambda
        self.deg_logit = deg_logit

    def forward(self, input, target):
        N = input.size(0)  # number of samples
        C = input.size(1)  # number of classes
        Mask = torch.zeros_like(input, requires_grad=False)
        Mask[range(N), target] = 1

        if self.deg_logit is not None:
            input = input - self.deg_logit * Mask

        loss = F.cross_entropy(input, target, weight=self.weight,
                               ignore_index=self.ignore_index, reduction=self.reduction)

        X = input - 1.e6 * Mask  # [N x C], excluding the target class
        reg = 0.5 * ((F.softmax(X, dim=1) - 1.0 / (C - 1)) * F.log_softmax(X, dim=1) * (1.0 - Mask)).sum(dim=1)
        if self.reduction == 'sum':
            reg = reg.sum()
        elif self.reduction == 'mean':
            reg = reg.mean()
        elif self.reduction == 'none':
            reg = reg

        return loss + self.reg_lambda * reg


def get_eval_res(prediction, label):
    flg_idx = label.cpu().detach().numpy().max(axis=(1, 2))
    pixel_idx = label.cpu().detach().numpy().sum(axis=(1, 2)).tolist()
    classification_score = F.softmax(prediction, dim=1)[:, 1, :, :].detach().cpu().numpy().max(axis=(1, 2)).tolist()

    dice = list(dice_coefficient(prediction, label, reduction='none').detach().cpu().numpy())
    ja = list(jaccard_coefficient(prediction, label, reduction='none').detach().cpu().numpy())
    hd = list(hausdorff_distance(prediction, label, reduction='none'))
    pr = list(precision(prediction, label, reduction='none').detach().cpu().numpy())
    sn = list(sensitivity(prediction, label, reduction='none').detach().cpu().numpy())

    return dice, ja, hd, pr, sn, flg_idx, pixel_idx, classification_score


def focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Arguments:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples or -1 for ignore. Default = 0.25
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = F.softmax(inputs, dim=1)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets.float(), reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def dice_single_image(prediction, target, reduction='mean', smooth=1):
    """

    :param prediction: [H, W]
    :param target: [H, W]
    :param reduction: 'mean', 'sum', None
    :param smooth:
    :return:
    """
    num_dims = len(prediction.shape)
    sum_idx = tuple(np.linspace(0, num_dims-1, num_dims, dtype=np.int64))

    numerator = (prediction * target).sum(axis=sum_idx)
    denominator = prediction.sum(axis=sum_idx) + target.sum(axis=sum_idx)

    dice = (2 * numerator + smooth) / (denominator + smooth)

    if reduction == 'mean':
        return dice.mean()
    elif reduction == 'sum':
        return dice.sum()
    elif reduction == 'none':
        return dice
    return None


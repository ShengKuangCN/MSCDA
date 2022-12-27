from torch.nn import init
import torch
import torch.nn as nn
import functools
from torch.optim import lr_scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            # elif init_type == 'no':
            #     pass
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', gpu_ids=[], pretrain=False):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
        if pretrain:
            ckpt_data = torch.load(pretrain, map_location='cuda:0')
            net.module.load_state_dict(ckpt_data['state_dict'])
            print(pretrain)
    if init_type and not pretrain:
        init_weights(net, init_type)
    return net


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def create_model(cfg, gpu_ids):
    if cfg.name.lower() == 'unet':
        from network.backbone.UNet import UnetGenerator
        net = UnetGenerator(cfg.input_nc, cfg.output_nc, cfg.num_downs, cfg.ngf, get_norm_layer(cfg.norm_layer),
                            cfg.use_dropout, cfg.req_feat)
    elif cfg.name.lower() == 'resnet50':
        from network.backbone.ResNet import ResNet50
        net = ResNet50(input_nc=cfg.input_nc, output_nc=cfg.output_nc, sync_bn=cfg.sync_bn,
                       replace_stride_with_dilation=cfg.dilation, dropout=cfg.dropout)
    elif cfg.name.lower() == 'resnet101':
        from network.backbone.ResNet import ResNet101
        net = ResNet101(input_nc=cfg.input_nc, output_nc=cfg.output_nc, sync_bn=cfg.sync_bn,
                        replace_stride_with_dilation=cfg.dilation, dropout=cfg.dropout)
    elif cfg.name.lower() == 'head':
        from network.ProjHead import ProjectionHead
        net = ProjectionHead(input_nc=cfg.input_nc, output_nc=cfg.output_nc)
    elif cfg.name.lower() == 'deeplab':
        from network.Deeplab import Deeplab
        net = Deeplab(backbone=cfg.backbone, input_nc=cfg.input_nc, output_nc=cfg.output_nc,
                      num_classes=cfg.num_classes, freeze_bn=cfg.freeze_bn, sync_bn=cfg.sync_bn, dropout=cfg.dropout)
    elif cfg.name.lower() == 'deeplab_contrast':
        from network.Deeplab import DeeplabContrast
        net = DeeplabContrast(backbone=cfg.backbone, input_nc=cfg.input_nc, output_nc=cfg.output_nc,
                              embed_nc=cfg.embed_nc, num_classes=cfg.num_classes, freeze_bn=cfg.freeze_bn,
                              sync_bn=cfg.sync_bn, dropout=cfg.dropout)
    else:
        raise NotImplementedError('network [%s] not implemented.' % cfg.name)

    net.__setattr__('name', cfg.name)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()

    print('[Network {}] Total number of parameters : {:.3} M, initialized with {}'.format(cfg.name, num_params / 1e6,
                                                                                          cfg.init_type))
    if not hasattr(cfg, 'pretrain') or not cfg.pretrain:
        return init_net(net, init_type=cfg.init_type, gpu_ids=gpu_ids)
    else:
        return init_net(net, init_type=None, gpu_ids=gpu_ids, pretrain=cfg.pretrain)


def get_scheduler(optimizer, cfg):
    if cfg.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + cfg.epoch_count - cfg.niter) / float(cfg.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif cfg.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.lr_decay_iters, gamma=0.1)
    elif cfg.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif cfg.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', cfg.lr_policy)
    return scheduler

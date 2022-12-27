import torch
import torch.nn as nn
import torch.nn.functional as F
from network.backbone.ResNet import ResNet50, ResNet101
from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from network.ProjHead import ProjectionHead


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        # self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, input_nc, output_nc, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        # if backbone == 'drn':
        #     input_nc = 512
        # elif backbone == 'mobilenet':
        #     input_nc = 320
        # else:
        #     input_nc = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(input_nc, output_nc, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(input_nc, output_nc, 3, padding=dilations[1], dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(input_nc, output_nc, 3, padding=dilations[2], dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(input_nc, output_nc, 3, padding=dilations[3], dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(input_nc, output_nc, 1, stride=1, bias=False),
                                             BatchNorm(output_nc),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(5 * output_nc, output_nc, 1, bias=False)
        self.bn1 = BatchNorm(output_nc)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class DeeplabDecoder(nn.Module):
    def __init__(self, num_classes, low_level_input_nc, low_level_output_nc, input_nc, fusion_nc, BatchNorm, sync_bn=False):
        super().__init__()
        # batch norm type
        if BatchNorm:
            pass
        elif sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.conv1 = nn.Conv2d(low_level_input_nc, low_level_output_nc, 1, bias=False)
        self.bn1 = BatchNorm(low_level_output_nc)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(
            nn.Conv2d(low_level_output_nc + input_nc, fusion_nc, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(fusion_nc),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(fusion_nc, fusion_nc, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm(fusion_nc),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(fusion_nc, num_classes, kernel_size=1, stride=1)
        )

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class DeeplabContrast(nn.Module):
    def __init__(self, backbone, input_nc, output_nc, embed_nc, num_classes=2, freeze_bn=False, sync_bn=False, dropout=0):
        super(DeeplabContrast, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        # batch norm type
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # initialize backbone
        if backbone == 'resnet50':
            self.backbone = ResNet50(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                     replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        elif backbone == 'resnet101':
            self.backbone = ResNet101(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                      replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        else:
            raise NotImplementedError('Backbone not implemented')

        self.aspp = ASPP(input_nc=4 * output_nc[-1], output_nc=output_nc[-1], output_stride=16, BatchNorm=BatchNorm)
        self.decoder = DeeplabDecoder(num_classes=num_classes, low_level_input_nc=output_nc[2], low_level_output_nc=48,
                                      input_nc=output_nc[-1], fusion_nc=output_nc[-1], BatchNorm=BatchNorm)
        self.projection_head = ProjectionHead(input_nc=4 * output_nc[-1], output_nc=embed_nc)
        self.freeze_bn = freeze_bn

    def forward(self, input, req_highdim_feat=False):
        x, low_level_feat = self.backbone(input)
        if req_highdim_feat:
            feat_highdim = x
        feat = self.projection_head(x)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if req_highdim_feat:
            return feat, x, feat_highdim
        else:
            return feat, x  # feature map, segmentation logits

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


class DeeplabContrast2(nn.Module):
    def __init__(self, backbone, input_nc, output_nc, embed_nc, num_classes=2, freeze_bn=False, sync_bn=False, dropout=0):
        super(DeeplabContrast2, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        # batch norm type
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # initialize backbone
        if backbone == 'resnet50':
            self.backbone = ResNet50(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                     replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        elif backbone == 'resnet101':
            self.backbone = ResNet101(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                      replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        else:
            raise NotImplementedError('Backbone not implemented')

        self.aspp = ASPP(input_nc=4 * output_nc[-1], output_nc=4 * output_nc[-1], output_stride=16, BatchNorm=BatchNorm)
        self.decoder = DeeplabDecoder(num_classes=num_classes, low_level_input_nc=output_nc[2], low_level_output_nc=48,
                                      input_nc=4 * output_nc[-1], fusion_nc=output_nc[-1], BatchNorm=BatchNorm)
        self.projection_head = ProjectionHead(input_nc=4 * output_nc[-1], output_nc=embed_nc)
        self.freeze_bn = freeze_bn

    def forward(self, input, req_highdim_feat=False):
        x, low_level_feat = self.backbone(input)
        if req_highdim_feat:
            feat_highdim = x
        x = self.aspp(x)
        feat = self.projection_head(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if req_highdim_feat:
            return feat, x, feat_highdim
        else:
            return feat, x  # feature map, segmentation logits

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


class Deeplab(nn.Module):
    def __init__(self, backbone, input_nc, output_nc, num_classes=2, freeze_bn=False, sync_bn=False, dropout=0):
        super(Deeplab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        # batch norm type
        if sync_bn:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        # initialize backbone
        if backbone == 'resnet50':
            self.backbone = ResNet50(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                     replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        elif backbone == 'resnet101':
            self.backbone = ResNet101(input_nc=input_nc, output_nc=output_nc, sync_bn=sync_bn,
                                      replace_stride_with_dilation=[False, 2, 4], dropout=dropout)
        else:
            raise NotImplementedError('Backbone not implemented')

        self.aspp = ASPP(input_nc=4 * output_nc[-1], output_nc=output_nc[-1], output_stride=16, BatchNorm=BatchNorm)
        self.decoder = DeeplabDecoder(num_classes=num_classes, low_level_input_nc=output_nc[2], low_level_output_nc=48,
                                      input_nc=output_nc[-1], fusion_nc=output_nc[-1], BatchNorm=BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input, req_highdim_feat=False):
        x, low_level_feat = self.backbone(input)
        if req_highdim_feat:
            feat_highdim = x
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        if req_highdim_feat:
            return x, feat_highdim
        else:
            return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


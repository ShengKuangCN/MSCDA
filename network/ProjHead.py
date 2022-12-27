import torch.nn as nn
import torch.nn.functional as F
from network.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
"""Reference: https://github.com/PatrickHua/SimSiam/blob/main/models/simsiam.py"""


class ProjectionHead(nn.Module):
    def __init__(self, input_nc, output_nc=256, proj='convmlp'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(input_nc, output_nc, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(input_nc, input_nc, kernel_size=1),
                # nn.SyncBatchNorm(input_nc),
                nn.BatchNorm2d(input_nc),
                nn.ReLU(),
                nn.Conv2d(input_nc, output_nc, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    import os
    import torch
    from torchsummary import summary
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    net = ProjectionHead(input_nc=512, output_nc=128)
    net.cuda()
    summary(net, input_size=(512, 32, 32))
    x = torch.randn((2, 512, 32, 32)).cuda()
    y = net(x)
    print(y.shape)

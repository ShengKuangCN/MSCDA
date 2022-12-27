import torch
import torch.nn as nn
import functools


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, req_feat=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
            req_feat        -- require feature map from bottleneck
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, req_feat=req_feat)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, req_feat=req_feat)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, req_feat=req_feat)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, req_feat=req_feat)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, req_feat=req_feat)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, req_feat=req_feat)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock2(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 req_feat=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock2, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        maxpool = nn.MaxPool2d(kernel_size=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            down = [downconv, maxpool]
            up = [uprelu, upconv, uprelu]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv, maxpool]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv, maxpool, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [nn.Dropout(use_dropout)] + [submodule] + up + [nn.Dropout(use_dropout)]
            else:
                model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            # print(x.shape, self.model(x).shape)
            return torch.cat([x, self.model(x)], 1)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 req_feat=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        self.req_feat = req_feat
        self.submodule = submodule
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)
        maxpool = nn.MaxPool2d(kernel_size=2)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.LeakyReLU(0.2, True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0)
            down = [downconv, maxpool]
            up = [uprelu, upconv, uprelu]
            model_pre = down
            model_sub = [submodule]
            model_post = up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv, maxpool]
            up = [uprelu, upconv, upnorm]
            model_pre = down
            model_post = up

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=2, stride=2,
                                        padding=0, bias=use_bias)
            down = [downrelu, downconv, maxpool, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model_pre = down + [nn.Dropout(use_dropout)]
                model_sub = [submodule]
                model_post = up + [nn.Dropout(use_dropout)]
            else:
                model_pre = down
                model_sub = [submodule]
                model_post = up

        self.model_pre = nn.Sequential(*model_pre)
        if not self.innermost:
            self.model_sub = nn.Sequential(*model_sub)
        self.model_post = nn.Sequential(*model_post)

    def forward(self, x):
        if self.req_feat:
            if self.outermost:
                y = self.model_pre(x)
                z, feat = self.model_sub(y)
                return self.model_post(z), feat + [y]
            elif self.innermost:   # add skip connections
                y = self.model_pre(x)
                return torch.cat([x, self.model_post(y)], 1), [y]
            else:
                y = self.model_pre(x)
                z, feat = self.model_sub(y)
                return torch.cat([x, self.model_post(z)], 1), feat + [y]
        else:
            if self.outermost:
                y = self.model_pre(x)
                z = self.model_sub(y)
                return self.model_post(z)
            elif self.innermost:   # add skip connections
                y = self.model_pre(x)
                return torch.cat([x, self.model_post(y)], 1)
            else:
                y = self.model_pre(x)
                z = self.model_sub(y)
                return torch.cat([x, self.model_post(z)], 1)


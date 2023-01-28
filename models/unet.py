import os
import torch
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Dropout3d, GroupNorm
from torch.nn import ReLU, Sigmoid
from monai.losses import DiceLoss
from seg_baseline import SegBaseline

class UNet(Module):
    """
    Class with implementation of methods needed to perform segmentation using Unet model.
    
    Args:
        unet_params (dict): parameters of monai unet.
            - num_channels (int): number of input channels.
            - num_channels_out (int): number of output channels.
            - feat_channels (list): channels in network.
            - residual (str): deafault = 'conv'
        save_path (str): path to save model.
        loss_function (str): only 'dice' avaiable for now.
        optimizer (str): only 'adam' available for now.
        epochs (int): number of epochs to train model.
    """
    def __init__(self, unet_params: dict, save_path: str, loss_function: str='dice', optimizer: str='adam', epochs: int=200) -> None:
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals
        super(UNet, self).__init__()

        num_channels = unet_params['num_channels']
        num_channels_out = unet_params['num_channels_out']
        feat_channels = unet_params['feat_channels']
        residual = unet_params['residual']

        self.save_path = save_path
        self.epochs = epochs
        
        if loss_function=='dice':
            self.loss_function = DiceLoss(sigmoid=True)
        if optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())

        # Encoder downsamplers
        self.pool1 = MaxPool3d((2, 2, 2))
        self.pool2 = MaxPool3d((2, 2, 2))
        self.pool3 = MaxPool3d((2, 2, 2))
        self.pool4 = MaxPool3d((2, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels_out, kernel_size=1, stride=1, padding=0, bias=True)

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # Encoder part
        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        x2 = self.conv_blk2(x_low1)

        x_low2 = self.pool2(x2)
        x3 = self.conv_blk3(x_low2)

        x_low3 = self.pool3(x3)
        x4 = self.conv_blk4(x_low3)

        x_low4 = self.pool4(x4)
        base = self.conv_blk5(x_low4)

        # Decoder part

        d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        d_high4 = self.dec_conv_blk4(d4)

        d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        d_high3 = self.dec_conv_blk3(d3)
        d_high3 = Dropout3d(p=0.5)(d_high3)

        d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
        d_high2 = self.dec_conv_blk2(d2)
        d_high2 = Dropout3d(p=0.5)(d_high2)

        d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
        d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)

        return seg

    def run_UNet(self, root_path: str) -> None:
        """
        Function starting segmentation training with unet monai model.

        Args:
            root_path (str): path to directory with csv files.
        """
        segmentation = SegBaseline(root_path)
        segmentation.training(self.forward, self.save_path, self.loss_function, self.optimizer, self.epochs)

class Conv3D_Block(Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            GroupNorm(out_feat, out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel, stride=stride, padding=padding, bias=True),
            GroupNorm(out_feat, out_feat),
            ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Deconv3D_Block(Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(kernel, kernel, kernel), stride=(stride, stride, stride), padding=(padding, padding, padding), output_padding=1, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)

class ChannelPool3d(AvgPool1d):
    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)

if __name__=='__main__':
    root_path = os.path.abspath(os.getcwd())
    unet_params = {'num_channels': 2,
                    'num_channels_out': 1,
                    'feat_channels': [4, 8, 16, 32, 64],
                    'residual': 'conv'}
    unet_monai = UNet(unet_params, 'save_path')
    unet_monai.run_UNet('root_path')
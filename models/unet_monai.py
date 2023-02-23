import os
import sys
import torch 
import pandas as pd

sys.path.insert(0, "C:/Users/danon/Desktop/Whole-body-segmentation")
from monai.networks.nets import UNet 
from monai.losses import DiceLoss
from seg_baseline import SegBaseline



class UnetMonai():
    """
    Class with implementation of methods needed to perform segmentation using built-in Unet Monai model.

    Args:
        unet_params (dict): parameters of monai unet
        save_path (str): path to save model
        loss_function (str): only 'dice' avaiable for now
        optimizer (str): only 'adam' available for now
        epochs (int): number of epochs to train model 
    """
    def __init__(self, unet_params: dict, save_path: str, loss_function: str='dice', optimizer: str='adam', epochs: int=20) -> None:
        super().__init__()
        self.model = UNet(
            spatial_dims=unet_params['spatial_dims'],       
            in_channels=unet_params['in_channels'],         
            out_channels=unet_params['out_channels'],       
            channels=unet_params['channels'],               
            strides=unet_params['strides'],                 
            num_res_units=unet_params['num_res_units'],     
        )
        self.unet_params = unet_params
        self.save_path = save_path
        self.epochs = epochs
        
        if loss_function=='dice':
            self.loss_function = DiceLoss(sigmoid=True)
        if optimizer=='adam':
            self.optimizer = torch.optim.Adam(self.model.parameters())

    def run_UnetMonai(self, root_path: str, checkpoint=None) -> None:
        """
        Function starting segmentation training with unet monai model.

        Args:
            root_path (str): path to directory with csv files.
        """
        segmentation = SegBaseline(root_path)
        segmentation.training(self.model, self.save_path, self.loss_function, self.optimizer, self.epochs, 'monai', checkpoint)

if __name__=='__main__':
    root_path = os.path.abspath(os.getcwd())
    unet_params = {'spatial_dims': 3,
                    'in_channels': 2,
                    'out_channels': 1,
                    'channels': (16, 32, 64, 128, 256),
                    'strides': (2, 2, 2, 2),
                    'num_res_units': 2}
    # unet_monai = UnetMonai(unet_params, 'save_path')
    # unet_monai.run_UnetMonai('C:/Users/danon/Desktop/Whole-body-segmentation')

    path = 'model_path'
    checkpoint = torch.load(path, map_location=torch.device('cpu'))

    unet_monai = UnetMonai(unet_params, 'save_path')
    unet_monai.run_UnetMonai(os.getcwd(), checkpoint)



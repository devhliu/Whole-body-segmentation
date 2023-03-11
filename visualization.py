import torch
import matplotlib.pyplot as plt
from seg_baseline import SegBaseline
from monai.networks.layers.factories import Norm

class Visualization():
    def __init__(self, root_path, model=None, model_path=None, loader='train', axis='axial'):
        seg_base = SegBaseline(root_path)
        self.train_loader = seg_base.load_dataloaders()[0]
        self.val_loader = seg_base.load_dataloaders()[1]
        self.test_loader = seg_base.load_dataloaders()[2]

        loader = self.train_loader
        if loader == 'val':
            self.loader = self.val_loader
        elif loader == 'test':
            self.loader = self.test_loader

        image, self.label = next(iter(loader))
        self.image = image.squeeze(0).permute(0,4,1,2,3)

        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        
        output = model(self.image)
        self.output = torch.sigmoid(output)

    def show_CT(self, slice):
        plt.imshow(self.image[:,0,:,:,:].squeeze(0).squeeze(0).detach().numpy()[slice, :, :], cmap='gray')
        plt.axis('off')
        plt.title('CT')

    def show_PET(self, slice):
        plt.imshow(self.image[:,1,:,:,:].squeeze(0).squeeze(0).detach().numpy()[slice, :, :], cmap='gray')
        plt.axis('off')
        plt.title('PET')

    def show_mask(self, slice):
        plt.imshow(self.label.squeeze(0).squeeze(0).detach().numpy()[slice, :, :], cmap='gray')
        plt.axis('off')
        plt.title('Mask')
    
    def show_output(self, slice):
        self.output[self.output > 0.5] = 1
        self.output[self.output <= 0.5] = 0

        plt.imshow(self.output.squeeze(0).squeeze(0).detach().numpy()[slice, :, :], cmap='gray')
        plt.axis('off')
        plt.title('Output')

    def full_visualization(self, slices=100):
        for i in range(slices):
            plt.figure()
            plt.subplot(1,4,1)
            self.show_CT(i)
            plt.subplot(1,4,2)
            self.show_PET(i)
            plt.subplot(1,4,3)
            self.show_mask(i)
            plt.subplot(1,4,4)
            self.show_output(i)
            plt.show()
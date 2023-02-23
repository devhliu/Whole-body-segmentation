import os
import torch
import pandas as pd
from tqdm import tqdm
from datetime import date
from eval_functions import count, dc, hd95
from dataloader_tio import AutopetDataloaderTio

class SegBaseline():
    """
    Class with implementation of methods needed to load data from csv files
    """
    def __init__(self, root_path: str) -> None:
        self.root_path = root_path

    def load_datasets(self, root_path: str):
        """
        Load data from csv files and preprocess by AutopeDataloaderTio class.

        Args:
            root_path (str): path to directory with csv files.
        
        Returns:
            train_dataset, val_dataset, test_dataset: dataloader_tio.AutopetDataloaderTio (image, label).
        """
        train_data = pd.read_csv(os.path.join(root_path, 'data_csv/train_dataset.csv'))
        val_data = pd.read_csv(os.path.join(root_path, 'data_csv/val_dataset.csv'))
        test_data = pd.read_csv(os.path.join(root_path, 'data_csv/test_dataset.csv'))

        ct_images_tr = train_data['CT']
        pet_images_tr = train_data['PET']
        suv_images_tr = train_data['SUV']
        labels_tr = train_data['MASKS']

        ct_images_val = val_data['CT']
        pet_images_val = val_data['PET']
        suv_images_val = val_data['SUV']
        labels_val = val_data['MASKS']

        ct_images_test = test_data['CT']
        pet_images_test = test_data['PET']
        suv_images_test = test_data['SUV']
        labels_test = test_data['MASKS']

        train_dataset = AutopetDataloaderTio(ct_images_tr, pet_images_tr, suv_images_tr, labels_tr)
        val_dataset = AutopetDataloaderTio(ct_images_val, pet_images_val, suv_images_val, labels_val)
        test_dataset = AutopetDataloaderTio(ct_images_test, pet_images_test, suv_images_test, labels_test)

        return train_dataset, val_dataset, test_dataset
    
    def load_dataloaders(self):
        """
        Load preprocessed datasets and process by troch dataloader.

        Returns:
            train_loader, val_loader, test_loader: torch.utils.data.dataloader.DataLoader (image, label)
        """
        train_dataset, val_dataset, test_dataset = self.load_datasets(self.root_path)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        return train_loader, val_loader, test_loader

    def training(self, model, save_path, loss_function, optimizer, epochs, name, checkpoint=None):
        device = torch.device("cpu")
        model = model.to(device)

        train_loader, val_loader, _ = self.load_dataloaders()

        training_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)

        current_epoch = 0
        prev_epochs = 0
        loss_history = []
        val_loss_history = []
        best_loss = float("inf")

        if checkpoint != None:
            loss_history = checkpoint['train_loss']
            val_loss_history = checkpoint['val_loss']
            best_loss = checkpoint['best_loss']
            current_epoch = checkpoint['epochs']

        now = date.today()
        for epoch in range(epochs):
            current_epoch += 1
            train_loss = 0.0
            val_loss = 0.0

            for image, label in tqdm(train_loader):
                image = image.squeeze(0)
                image = image.permute(0,4,1,2,3)
                image = image.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = model(image)

                loss = loss_function(pred, label)
                loss.backward()

                optimizer.step()
                train_loss += loss.item()

            loss_history.append(train_loss / training_size)
                
            for image, label in tqdm(val_loader):
                image = image.squeeze(0)
                image = image.permute(0,4,1,2,3)
                image = image.to(device)
                label = label.to(device)
                
                pred = model(image)
                loss = loss_function(pred, label)

                val_loss += loss.item()

            val_loss_history.append(val_loss / val_size)

            if (epoch+1)%4==0:
                print('Val Loss', val_loss / val_size)
                # model.eval()
                # with torch.no_grad():
                #     dc1, hd = count(model, val_loader, dc, hd95)

                # model.train()
                # print('\nDC after ' + str(epoch) + ' epochs: ' + str(dc1))
                # print('HD after ' + str(epoch) + ' epochs: ' + str(hd))

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_history,
                    'val_loss': val_loss_history,
                    'best_loss': best_loss,
                    'epochs': current_epoch,
                    }, os.path.join(save_path, f'{name}-{now}-epochs-' + str(current_epoch) + '.pt'))

            if val_loss / val_size < best_loss:
                best_loss = val_loss / val_size

                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': loss_history,
                    'val_loss': val_loss_history,
                    }, os.path.join(save_path, f'{name}-{now}-best.pt'))

        
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': loss_history,
            'val_loss': val_loss_history,
            'best_loss': best_loss,
            'epochs': current_epoch,
            }, os.path.join(save_path, f'{name}-{now}-epochs-' + str(current_epoch) + '.pt'))

    def evaluation(self, model, save_model) -> None:
        """
        Function evaluating results.

        Args:
            model: network architecture.
            save_model: saved model to evaluate.
        """
        checkpoint = torch.load(save_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        _, _, test_loader = self.load_dataloaders()
        count(model, test_loader, dc, hd95)
        model.train()


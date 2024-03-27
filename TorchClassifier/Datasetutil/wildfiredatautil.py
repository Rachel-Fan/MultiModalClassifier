import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from PIL import Image
import pandas as pd

class CustomDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, 'images', self.dataframe.iloc[idx, 0])
        image = Image.open(img_name)
        label = torch.tensor(self.dataframe.iloc[idx, 1:].tolist(), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label

class WildfireDatasetLoader:
    def __init__(self, name, path, valid_size=0.2, batch_size=32, num_workers=0):
        self.name = name
        self.path = path
        self.valid_size = valid_size
        self.batch_size = batch_size
        self.num_workers = num_workers

    def load(self):
        # Step 1: Data Preparation
        train_data = pd.read_csv(os.path.join(self.path, self.name, 'train', '_classes.csv'))  # Update the path
        valid_data = pd.read_csv(os.path.join(self.path, self.name, 'valid', '_classes.csv'))  # Update the path
        test_data = pd.read_csv(os.path.join(self.path, self.name, 'test', '_classes.csv'))  # Update the path

        # Step 2: Data Loading
        custom_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        custom_train_dataset = CustomDataset(train_data, root_dir=os.path.join(self.path, self.name, 'train'), transform=custom_transform)
        custom_valid_dataset = CustomDataset(valid_data, root_dir=os.path.join(self.path, self.name, 'valid'), transform=custom_transform)
        custom_test_dataset = CustomDataset(test_data, root_dir=os.path.join(self.path, self.name, 'test'), transform=custom_transform)

        # Prepare data loaders
        train_loader = torch.utils.data.DataLoader(custom_train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        valid_loader = torch.utils.data.DataLoader(custom_valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(custom_test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        # Obtain one batch of training images
        dataiter = iter(train_loader)
        images, labels = next(dataiter)
        imagetensorshape = list(images.shape)  # torch.Size to python list
        imageshape = imagetensorshape[1:]

        class_names = custom_train_dataset.classes
        print('Number of classes: ', len(class_names))

        return {
            'train': train_loader,
            'val': valid_loader,
            'test': test_loader
        }, {
            'train': len(custom_train_dataset),
            'val': len(custom_valid_dataset),
            'test': len(custom_test_dataset)
        }, class_names, imageshape

# Example usage:
# loader = WildfireDatasetLoader('wildfire', 'path_to_data_folder')
# dataloaders, dataset_sizes, class_names, imageshape = loader.load()

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np
import random

import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms


class FaceDataset(Dataset):
    label2classid = {
        "Live Face": 5,
        "Cutouts": 4,
        "Replay": 4,
        "Print": 4,
        "Face-Swap": 3,
        "Video-Driven": 2,
        "Attibute-Edit": 2,
        "Pixcel-Level": 1,
        "Semantic-Leve": 0,
    }
    def __init__(self, csv_file, label2classid=None, transform=None, is_train=True):
        self.data_df = pd.read_csv(csv_file)
       
        self.data_folder = os.path.dirname(csv_file)
        self.transform = transform
        if label2classid is not None:
            self.label2classid = label2classid
        self.data_df['class_id'] = self.data_df['sublabel'].apply(lambda x: self.label2classid.get(x, -1))

        if self.transform is None:
            if is_train:
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop((224, 224), scale=(0.65, 1)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                ])
            else:  # val or test
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)), 
                    transforms.ToTensor(),
                ])

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_name = self.data_df.iloc[idx]['filename']
        img_path = os.path.join(self.data_folder, img_name)

        sub_label = self.data_df.iloc[idx]['sublabel'] 
        image = Image.open(img_path).convert('RGB')
        class_id = self.data_df.iloc[idx]['class_id'] 
        if self.transform is not None:
            image = self.transform(image)
        
        return image, class_id, sub_label, img_name

def worker_init_fn(worker_id):
    """
    为每个worker设置随机种子，确保数据加载的随机性是可控的
    """
    worker_seed = torch.initial_seed() % 2**32
    print("worker_seed:", worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataloader(csv_file, is_train, batch_size=32, num_workers=4, seed=42, oversampling=False, **kwargs):
    

    face_dataset = FaceDataset(csv_file, is_train=is_train)
    
    g = torch.Generator()
    g.manual_seed(seed)
    # Add oversampling for classes 4 and 5 during training
    if oversampling:
        pass
        # Get all indices for each class
        num_classes = len(face_dataset.label2classid.values())
        class_sample_count = np.array([np.sum(face_dataset.data_df['class_id'] == i) for i in range(num_classes)])
        weights = 1. / class_sample_count
        sample_weights = np.array([weights[t] for t in face_dataset.data_df['class_id']])
        print("sample_weights:", sample_weights)
        # add more weight to classes 4 and 5 (live face and physical face)
        factor = 1.2
        for idx, label in enumerate(face_dataset.data_df['class_id']):
            if label in [4, 5]:
                sample_weights[idx] *= factor

        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), generator=g)
        kwargs['sampler'] = sampler
        kwargs.pop('shuffle', None)

    dataloader = DataLoader(
        face_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        **kwargs
    )
    
    
    return dataloader, face_dataset

class InputNormalize(nn.Module):
    '''
    A module (custom layer) for normalizing the input to have a fixed
    mean and standard deviation (user-specified).
    '''
    def __init__(self, new_mean=None, new_std=None, device=torch.device('cuda')):
        super(InputNormalize, self).__init__()
        if new_mean is None or new_std is None:
            new_mean = torch.tensor([0.485, 0.456, 0.406])
            new_std = torch.tensor([0.229, 0.224, 0.225])
        new_std = new_std[..., None, None].to(device)
        new_mean = new_mean[..., None, None].to(device)

        # To prevent the updates the mean, std
        self.register_buffer("new_mean", new_mean)
        self.register_buffer("new_std", new_std)

    def forward(self, x):
        x = torch.clamp(x, 0, 1)
        x_normalized = (x - self.new_mean)/self.new_std
        return x_normalized

def build_finetune_dataloader(csv_file, is_train=True, batch_size=32, num_workers=4, seed=42, **kwargs):
    label2classid = {
        "Live Face": 4,
        "Cutouts": 3,
        "Replay": 3,
        "Print": 3,
        "Face-Swap": 2,
        "Video-Driven": 1,
        "Attibute-Edit": 1,
        # "Pixcel-Level": 1,
        "Semantic-Leve": 0,
        "unknown": -1,
        'Plaster': -1,
    }
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    face_dataset = FaceDataset(csv_file, is_train=is_train, label2classid=label2classid, transform=transform)

    g = torch.Generator()
    g.manual_seed(seed)
    
    dataloader = DataLoader(
        face_dataset, 
        batch_size=batch_size, 
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=g,
        **kwargs
    )
    
    
    return dataloader, face_dataset
from pathlib import Path
import shutil
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CustomImageDataset(Dataset):
  def __init__(self, data_dir_path, transform=None, target_transform=None):
      self.img_labels, self.label_names = self.load_data(data_dir_path)
      self.transform = transform
      self.target_transform = target_transform

  def __len__(self):
      return len(self.img_labels)

  def load_data(self, data_dir_path):
    label_names = ['cats', 'chickens', 'dogs', 'horses', 'sheep']
    data = dict()

    for dir_path in data_dir_path.iterdir():
      label_name = dir_path.name
      assert label_name in label_names
      label = label_names.index(label_name)
      image_file_paths = dir_path.glob('*.*')

      for image_file_path in image_file_paths:
        data.setdefault('image_file_path', [])
        data['image_file_path'].append(str(image_file_path))
        data.setdefault('label', [])
        data['label'].append(label)

    img_labels = pd.DataFrame.from_dict(data)

    return img_labels, label_names

  def __getitem__(self, idx):
      img_path = self.img_labels.iloc[idx, 0]
      image = read_image(img_path)
      # image = Image.open(img_path)
      label = torch.tensor(self.img_labels.iloc[idx, 1])
      if self.transform:
          image = self.transform(image)
      if self.target_transform:
          label = self.target_transform(label)
      return image, label


transform = transforms.Compose(
    [
      transforms.ToPILImage(),
      transforms.Resize((320, 320)),
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

batch_size = 4

train_dataset = CustomImageDataset(data_root_dir_path / 'train', transform=transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

test_dataset = CustomImageDataset(data_root_dir_path / 'test', transform=transform)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = train_dataset.label_names

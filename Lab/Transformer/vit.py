from __future__ import print_function

import glob
from itertools import chain
import os
import random
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from linformer import Linformer
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

from vit_pytorch.efficient import ViT

dim=128
batch_size = 64
epochs = 2000 
lr = 0.0003
gamma = 1/dim ** 0.5
seed = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(seed)
    


device = 'cuda'

train_dir = '/data/sunhaoyu/Lab/Transformer/alz_data/train'
test_dir = '/data/sunhaoyu/Lab/Transformer/alz_data/test'


train_list = glob.glob(os.path.join(train_dir,'*.jpg'))
test_list = glob.glob(os.path.join(test_dir, '*.jpg'))

labels = [path.split('/')[-1].split('.')[0][0:7] for path in train_list]
#labels = [path.split('/')[-1].split('.')[0] for path in train_list]


train_list, valid_list = train_test_split(train_list, 
                                          
                                          test_size=0.2,
                                          stratify=labels,
                                          random_state=seed)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]
)

class AlzDataset(Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img = img.resize((224,224))
        img_transformed = self.transform(img)
        #label = img_path.split("/")[-1].split(".")[0]
        label = img_path.split("/")[-1].split(".")[0][0:7]
        #label = 1 if label == "dog" else 0
        #label = 0 if label == "Non" else 1
        
        if label == "VeryMil":
            label = 3
        if label == "MildDem":
            label = 2
        if label == "Moderat":
            label = 1
        if label == "NonDemn":
            label = 0
        

        return img_transformed, label
    
train_data = AlzDataset(train_list, transform=train_transforms)
valid_data = AlzDataset(valid_list, transform=test_transforms)
test_data = AlzDataset(test_list, transform=test_transforms)

train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True )
valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=True)

efficient_transformer = Linformer(
    dim=dim,
    seq_len=50,  # 7x7 patches + 1 cls-token
    depth=12,
    heads=8,
    k=128
).to(device)

model = ViT(
    dim=dim,
    image_size=224,
    patch_size=32,
    num_classes=4,
    transformer=efficient_transformer,
    channels=1,
).to(device)

# loss function
criterion = nn.CrossEntropyLoss()
# optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)
# scheduler
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []
training_history = []
best_val_accuracy = 0.0  # initialize the best validation accuracy
best_model_path = "/data/sunhaoyu/Lab/Transformer/linformer_model/best_model-2-4-5000.pt"

for epoch in range(epochs):
    epoch_loss = 0
    epoch_accuracy = 0
    
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()

        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
        
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0
        for data, label in valid_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(valid_loader)
            epoch_val_loss += val_loss / len(valid_loader)

        if epoch_val_accuracy > best_val_accuracy:  
            best_val_accuracy = epoch_val_accuracy  
            torch.save(model.state_dict(), best_model_path)
            
    train_loss_history.append(f"{epoch_loss:.4f}")
    
    train_acc_history.append(f"{epoch_accuracy:.4f}")
    val_loss_history.append(f"{epoch_val_loss:.4f}")
    val_acc_history.append(f"{epoch_val_accuracy:.4f}")
    training_history.append(f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n")
    with open('training_history.txt', 'w') as f:
        for item in training_history:
            
            f.write("%s\n" % item)
    print(
        f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n"
    )
    

train_loss_history = list(map(float, train_loss_history))
train_acc_history = list(map(float, train_acc_history))
val_loss_history = list(map(float, val_loss_history))
val_acc_history = list(map(float, val_acc_history))

print(train_loss_history)
print(train_acc_history)
print(val_loss_history)
print(val_acc_history)
print(f"max_val_acc : {best_val_accuracy}")

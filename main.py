import sklearn.model_selection
import torch.utils
import torch.utils.data
from models import cafenet
import dataset
import trainer

import torch
import torchvision
from torchvision import transforms
import numpy as np
import sklearn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folder_path = "./data/tma_01"

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness = 0.1, contrast = 0.1, saturation = 0.1, hue = 0.1),
    transforms.RandomRotation(degrees = 15),
    transforms.GaussianBlur(kernel_size = (3,3), sigma=(0.1,2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_set = dataset.KBSMC_dataset(folder_path=folder_path,transform=train_transforms)
test_set = dataset.KBSMC_dataset(folder_path=folder_path,transform=test_transforms)

indices = list(range(len(train_set)))
train_indices, test_indices = sklearn.model_selection.train_test_split(indices)
train_dataset = torch.utils.data.Subset(train_set, train_indices)
test_dataset = torch.utils.data.Subset(test_set,test_indices)

myModel = cafenet.CaFeNet(num_classes=4).to(device=device)

print(train_dataset)

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#criterion should not be here. it should be in the model.py
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)


myTrainer = trainer.Trainer(model=myModel, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, device=device)
print(f"Using device: {device}")



myTrainer.train(num_epochs=1)
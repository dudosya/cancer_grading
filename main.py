import torch.utils
import torch.utils.data
from models import cafenet
import dataset
import trainer

import torch
import torchvision
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

folder_path = "./data"
myDataset = dataset.KBSMC_dataset(folder_path=folder_path)
myModel = cafenet.CaFeNet(num_classes=4).to(device=device)

dataset_size = len(myDataset)
train_dataset_size = int(0.8*dataset_size)
test_dataset_size = dataset_size - train_dataset_size

train_dataset, test_dataset = torch.utils.data.random_split(myDataset, [train_dataset_size, test_dataset_size])

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

train_dataset.transform = train_transforms
test_dataset.transform = test_transforms

batch_size = 16

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#criterion should not be here. it should be in the model.py
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(myModel.parameters(), lr=0.001)


myTrainer = trainer.Trainer(model=myModel, train_loader=train_loader, test_loader=test_loader, criterion=criterion, optimizer=optimizer, device=device)
print(f"Using device: {device}")

myTrainer.train(num_epochs=1)
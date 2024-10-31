import torch
import os
import re
from PIL import Image
from torchvision import transforms

class KBSMC_dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, transform = None):
        self.folder_path = folder_path
        self.transform = transform
        self.__traverse__()
        
    def __getitem__(self, index):
        #returns a single data sample (tuple) given an index
        image = Image.open(self.img_paths[index])
        
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[index]
        
        return image,label
    
    def __len__(self):
        return len(self.labels)
    
    def __traverse__(self):
        #traverse the dataset and create imgs and labels lists
        self.img_paths = []
        self.labels = []
        
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', 'jpeg')):
                    img_path = os.path.join(root,file)
                    self.img_paths.append(img_path)
                    label = int(file.split('_')[-1].split('.')[0])
                    self.labels.append(label)
    
    


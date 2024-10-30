import torch
import os
import re
from PIL import Image
from torchvision import transforms

class KBSMC_dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.__traverse__()
        
    def __getitem__(self, index):
        #returns a single data sample (tuple) given an index
        image = Image.open(self.img_paths[index])
        basic_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        image = basic_transform(image)
        label = self.labels[index]
        
        return image,label
    
    def __len__(self):
        return len(self.labels)
    
    def __traverse__(self):
        #traverse the dataset and create imgs and labels lists
        self.img_paths = []
        self.labels = []
        
        # for root, _, files in os.walk(self.folder_path):
        #     for file in files:
        #         if file.endswith(('.png', '.jpg', 'jpeg')):
        #             img_path = os.path.join(root,file)
        #             self.img_paths.append(img_path)
        #             label = int(file.)
        
        
        for folder in os.listdir(self.folder_path):
            img_names = os.listdir(os.path.join(self.folder_path,folder))
            for img_name in img_names:
                self.img_paths.append(os.path.join(self.folder_path, folder, img_name))
                self.labels.append(int(re.split("_",img_name)[-1][0]))
    
    


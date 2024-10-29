import torch
import os
import re

class KBSMC_dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.__traverse__()
        
    def __getitem__(self, index):
        #returns a single data sample (tuple) given an index
        return self.img_paths[index], self.labels[index]
    
    def __len__(self):
        return len(self.labels)
    
    def __traverse__(self):
        #traverse the dataset and create imgs and labels lists
        self.img_paths = []
        self.labels = []
        
        for folder in os.listdir(self.folder_path):
            img_names = os.listdir(os.path.join(self.folder_path,folder))
            for img_name in img_names:
                self.img_paths.append(os.path.join(self.folder_path, folder, img_name))
                self.labels.append(re.split("_",img_name)[-1][0])
    
    


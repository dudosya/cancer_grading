import os
from collections import Counter
import torch
from torch.utils.data import Dataset
from PIL import Image

def print_freq_table_tma_folder():
    """
    prints out freq table like this:

    Label 0: 163
    Label 1: 2322
    Label 2: 4105
    Label 3: 1830

    """

    dir = './data/KBSMC_colon_tma_cancer_grading_512'
    label_counter = Counter()

    for folder in os.listdir(dir) :
        path = os.path.join(dir,folder)

        if os.path.isdir(path) and folder.startswith('tma'):
            #print(f"Processing folder: {folder}")

            for filename in os.listdir(path):
                if filename.endswith('.jpg'):
                    label = filename.split('_')[-1].split('.')[0]

                    if label.isdigit() and 0 <= int(label) <=3:
                        label_counter[int(label)] += 1

    print("Frequency Table: ")
    for label, count in sorted(label_counter.items()):
        print(f"Label {label}: {count}")


class RealCancerDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        """
        Initializes the dataset for real cancer grading images.

        Args:
            folder_path (str): Path to the dataset folder.
            transform (callable, optional): Optional transform to apply to the images.
        """
        super().__init__()
        self.folder_path = folder_path
        self.transform = transform

        # Class labels: 0 - BN, 1 - WD, 2 - MD, 3 - PD
        self.classes = ['BN', 'WD', 'MD', 'PD']
        self.num_classes = len(self.classes)

        # Load all image file paths
        self.image_paths = []
        self.labels = []

        # Traverse through the folder to get images and labels
        for root, _, files in os.walk(self.folder_path):
            for file in files:
                if file.endswith(('.png', '.jpg', '.jpeg')):  # Adjust for file types
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)

                    # Extract label from the file name (e.g., 'someid_sometext_0.jpg')
                    # Assuming label is in the filename like 'xxx_xxx_label.jpg'
                    label = int(file.split('_')[-1].split('.')[0])  # Extract the label part
                    self.labels.append(label)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Returns a tuple (image, label) for a given index.
        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            tuple: (image, label) where label is the class of the image.
        """
        # Load the image
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Extract the corresponding label
        label = self.labels[idx]

        # Apply transformation if provided
        if self.transform:
            image = self.transform(image)

        return image, label





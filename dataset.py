import os
from collections import Counter
import torch
from torch.utils.data import Dataset

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



# TODO: create dataset class?





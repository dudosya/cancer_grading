# import files
from models import *
import dataset
import trainer
import config as CONFIGURE
import wandb

#import libs
import torch
import torchvision
from torchvision import transforms
import numpy as np
import sklearn
import random
import sklearn.model_selection
import torch.utils
import torch.utils.data

def set_seed(seed=CONFIGURE.seed_num):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main():

    #Reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if CONFIGURE.wandb == True:
        wandb.init(
            project = CONFIGURE.wandb_project_name,
            name = CONFIGURE.wandb_run_name,
            tags = CONFIGURE.wandb_tags,
            monitor_gym = CONFIGURE.wandb_monitor_gym,
            #id = CONFIGURE.wandb_run_id,
            #resume = CONFIGURE.wandb_resume_run
        )
        print("wandb is ON")
        
    else:
        print("wandb is OFF")



    train_set = dataset.KBSMC_dataset(folder_path=CONFIGURE.folder_path,transform=CONFIGURE.train_transforms)
    test_set = dataset.KBSMC_dataset(folder_path=CONFIGURE.folder_path,transform=CONFIGURE.test_transforms)
    indices = list(range(len(train_set)))
    train_indices, test_indices = sklearn.model_selection.train_test_split(indices)
    train_dataset = torch.utils.data.Subset(train_set, train_indices)
    test_dataset = torch.utils.data.Subset(test_set,test_indices)

    myModel = GoogLeNetCustom_CBAM4(num_classes=CONFIGURE.num_classes, lr = CONFIGURE.learning_rate).to(device=device)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=CONFIGURE.batch_size,shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIGURE.batch_size, shuffle=False)

    myTrainer = trainer.Trainer(
        model=myModel, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        criterion=myModel.criterion, 
        optimizer=myModel.optimizer, 
        device=device)
    
    print(f"Using device: {device}")
    myModel.print_num_params()
    myTrainer.train(num_epochs=CONFIGURE.num_epochs)


if __name__ == "__main__":
    main()
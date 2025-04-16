from torchvision import transforms
import torch


#CONFIG
seed_num = 7
folder_path = "/dataset/KBSMC_colon_tma_cancer_grading_512"



train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees = 15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

test_transforms = transforms.Compose([
        transforms.Resize(256),       # Resize to a larger size
        transforms.CenterCrop(224),   # Crop the center to 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

learning_rate = 1e-3
num_classes = 4
batch_size = 64
num_epochs = 20

wandb = True
wandb_project_name = "16 April 2025"
wandb_run_name = "VGG16"
wandb_tags = ["normal_dataset"]
wandb_monitor_gym = True
#wandb_run_id = "xvio3t2p"
#wandb_resume_run = "allow" # "never" if dont wanna
        
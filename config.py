from torchvision import transforms


#CONFIG
seed_num = 7
folder_path = "./data"

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

learning_rate = 0.001
num_classes = 4
batch_size = 16
num_epochs = 20

wandb = True
wandb_project_name = "cancer_grading_ml_project"
wandb_run_name = "ViT-Base"
wandb_tags = ["ml_course"]
wandb_monitor_gym = True
#wandb_run_id = "xvio3t2p"
#wandb_resume_run = "allow" # "never" if dont wanna
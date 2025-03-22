from torchvision import transforms
import torch
import kornia.filters as filters

#CONFIG
seed_num = 7
folder_path = "./data"


class UnsharpMask(torch.nn.Module):
        def __init__(self, kernel_size, sigma, border_type = 'reflect'):
                super().__init__()
                self.kernel_size = kernel_size
                self.sigma = sigma
                self.border_type = border_type
        
        def forward(self,img):
                return filters.unsharp_mask(img.unsqueeze(0), self.kernel_size, self.sigma, self.border_type).squeeze(0)
        
        
class LaplacianFilter(torch.nn.Module):
        """
        Applies the Laplacian filter to an input image using Kornia.

        Args:
                kernel_size (int): Size of the Laplacian kernel (e.g., 3, 5, 7).  Must be odd.
                border_type (str, optional): Type of padding.  See Kornia's documentation for options.
                Defaults to 'reflect'.
        """
        def __init__(self, kernel_size, border_type='reflect'):
                super().__init__()
                if kernel_size % 2 == 0:
                        raise ValueError("Kernel size must be odd.")
                self.kernel_size = kernel_size
                self.border_type = border_type
                self.laplacian = filters.Laplacian(kernel_size, border_type)  # Use Kornia's Laplacian

        def forward(self, img):
                """
                Applies the Laplacian filter.

                Args:
                img (torch.Tensor): Input image tensor.  Should be of shape [C, H, W].

                Returns:
                torch.Tensor: The filtered image tensor, of shape [C, H, W].
                """
                # Kornia's filters expect a batch dimension, so unsqueeze and then squeeze.
                return self.laplacian(img.unsqueeze(0)).squeeze(0)



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

learning_rate = 0.001
num_classes = 4
batch_size = 16
num_epochs = 1

wandb = False
wandb_project_name = "4_march_2"
wandb_run_name = "CvT"
wandb_tags = ["normal_dataset"]
wandb_monitor_gym = True
#wandb_run_id = "xvio3t2p"
#wandb_resume_run = "allow" # "never" if dont wanna
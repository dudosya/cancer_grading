import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class VGG16(nn.Module):
    def __init__(self, num_classes=4, lr=0.001):
        super(VGG16, self).__init__()
        
        # Load the pre-trained VGG16 model with the recommended weights
        self.feature_extractor = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Replace the classifier's last layer to match the number of classes
        self.feature_extractor.classifier[6] = nn.Linear(4096, num_classes)
        
        # Define the criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        
    def forward(self, x, labels=None):
        # Forward pass through the model
        logits = self.feature_extractor(x)
        return logits, logits  # Returning logits twice to match expected output format
    
    def print_num_params(self):
        # Print the total number of trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")

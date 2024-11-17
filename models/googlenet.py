import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class GoogLeNet(nn.Module):
    def __init__(self, num_classes=4, lr=0.001):
        super(GoogLeNet, self).__init__()
        
        # Load the pre-trained GoogLeNet model with the recommended weights
        self.feature_extractor = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
        
        # Replace the classifier's last layer to match the number of classes
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features, num_classes)
        
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

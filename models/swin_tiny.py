import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import swin_t, Swin_T_Weights

class SwinTiny(nn.Module):
    def __init__(self, num_classes=4, lr=0.001):
        super(SwinTiny, self).__init__()
        
        # Load the pre-trained Swin-Tiny model with the recommended weights
        self.feature_extractor = swin_t(weights=Swin_T_Weights.DEFAULT)
        
        # Replace the classifier's head to match the number of classes
        self.feature_extractor.head = nn.Linear(self.feature_extractor.head.in_features, num_classes)
        
        # Define the criterion and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x, labels=None):
        # Forward pass through the model
        logits = self.feature_extractor(x)
        return logits, logits  # Returning logits twice to match expected output format
    
    def print_num_params(self):
        # Print the total number of trainable parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

class EfficientNetB1(nn.Module):
    """
    Wrapper class for EfficientNet-B1.

    Loads a pre-trained EfficientNet-B1 model, replaces the final classifier layer,
    and initializes loss criterion and optimizer.
    """
    def __init__(self, num_classes=4, lr=0.001):
        """
        Initializes the EfficientNetB1 model.

        Args:
            num_classes (int): The number of output classes for the final layer. Default is 4.
            lr (float): The learning rate for the optimizer. Default is 0.001.
        """
        super(EfficientNetB1, self).__init__()

        # Load the pre-trained EfficientNet-B1 model with recommended weights
        # Note: Using EfficientNet_B1_Weights.IMAGENET1K_V1 for consistency
        self.feature_extractor = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)

        # EfficientNet models in torchvision often have a 'classifier' attribute
        # which is an nn.Sequential. The last layer is typically the nn.Linear layer.
        # We need to replace this last layer.

        # Get the number of input features for the classifier's linear layer
        in_features = self.feature_extractor.classifier[-1].in_features

        # Replace the final layer (the nn.Linear layer within the classifier nn.Sequential)
        self.feature_extractor.classifier[-1] = nn.Linear(in_features, num_classes)

        # Define the criterion (loss function)
        self.criterion = nn.CrossEntropyLoss()

        # Define the optimizer
        # Note: Adam/AdamW is often preferred for EfficientNets, but using SGD
        #       to strictly match the GoogLeNet example structure.
        #       You might want to experiment with AdamW:
        #       self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x, labels=None):
        """
        Forward pass through the EfficientNet-B1 model.

        Args:
            x (torch.Tensor): The input tensor (batch of images).
            labels (torch.Tensor, optional): Labels for the input data. Not used
                                             in this forward pass implementation,
                                             but included for potential compatibility.
                                             Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the logits twice,
                                               matching the GoogLeNet example's output format.
        """
        # Forward pass through the modified EfficientNet model
        logits = self.feature_extractor(x)

        # Returning logits twice to match the expected output format of the example
        return logits, logits

    def print_num_params(self):
        """
        Calculates and prints the total number of trainable parameters in the model.
        """
        # Calculate the total number of parameters that require gradients
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # Convert to millions for readability
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")

# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    NUM_CLASSES = 10  # Example: 10 classes
    LEARNING_RATE = 0.005
    INPUT_SIZE = (240, 240) # EfficientNet-B1 default input size

    # Create the model instance
    model = EfficientNetB1(num_classes=NUM_CLASSES, lr=LEARNING_RATE)

    # Print the number of trainable parameters
    model.print_num_params() # Output will be around 6.6M for B1

    # Create a dummy input tensor
    # Batch size = 4, Channels = 3, Height = 240, Width = 240
    dummy_input = torch.randn(4, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    # Put model in evaluation mode for inference example (disables dropout, etc.)
    model.eval()

    # Perform a forward pass
    with torch.no_grad(): # Disable gradient calculation for inference
        logits1, logits2 = model(dummy_input)

    # Print output shapes
    print(f"Logits 1 shape: {logits1.shape}") # Should be [batch_size, num_classes] e.g., [4, 10]
    print(f"Logits 2 shape: {logits2.shape}") # Should be the same

    # You can access the criterion and optimizer like this:
    print(f"Criterion: {model.criterion}")
    print(f"Optimizer: {model.optimizer}")

    # Example of accessing parameters for the optimizer (if you were training)
    # optimizer = model.optimizer
    # loss = model.criterion(logits1, target_labels) # Assuming you have target_labels
    # loss.backward()
    # optimizer.step()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from attention_modules.CBAM import CBAM # Assuming original CBAM.py

class EfficientNetB1_single_CBAM(nn.Module):
    def __init__(self, num_classes=4, lr=0.001, reduction_ratio=16, spatial_kernel_size=7):
        super().__init__()
        self.base_model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)

        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool

        # Get original classifier details
        in_features = self.base_model.classifier[-1].in_features
        dropout_p = self.base_model.classifier[0].p

        # Determine channels BEFORE avgpool (output of self.features)
        # We need the channel count of the final conv layer in 'features'
        final_conv_channels = self._get_final_feature_channels(self.features)
        if final_conv_channels is None:
             # Fallback for B1 - it's 1280
             print("Warning: Could not determine final feature channels automatically, assuming 1280 for B1.")
             final_conv_channels = 1280

        # Add CBAM after the main features, before pooling
        self.cbam_block = CBAM(gate_channels=final_conv_channels,
                               reduction_ratio=reduction_ratio,
                               spatial_kernel_size=spatial_kernel_size)

        # Define the new classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(final_conv_channels, num_classes) # Input features are now final_conv_channels
        )

        self.criterion = nn.CrossEntropyLoss()
        # Strongly recommend AdamW and a LOW LR (e.g., 1e-4) for testing this
        self.optimizer = optim.AdamW(self.parameters(), lr=lr)
        # self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def _get_final_feature_channels(self, feature_module):
         """Tries to find output channels of the last conv/bn layer in features."""
         last_channel_defining_layer = None
         for layer in reversed(list(feature_module.modules())):
             if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                 last_channel_defining_layer = layer
                 break
         if last_channel_defining_layer:
              if isinstance(last_channel_defining_layer, nn.Conv2d):
                   return last_channel_defining_layer.out_channels
              elif isinstance(last_channel_defining_layer, nn.BatchNorm2d):
                   return last_channel_defining_layer.num_features
         return None # Indicate failure

    def forward(self, x, labels=None):
        x = self.features(x)
        x = self.cbam_block(x) # Apply CBAM here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits, logits # Or just logits

    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")


# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    NUM_CLASSES = 10  # Example: 10 classes
    LEARNING_RATE = 0.005 # NOTE: This LR might be too high for SGD, consider lower (e.g., 1e-3, 1e-4) or use AdamW
    INPUT_SIZE = (240, 240) # EfficientNet-B1 default input size

    # Create the model instance
    model = EfficientNetB1_single_CBAM(num_classes=NUM_CLASSES, lr=LEARNING_RATE)

    # Print the number of trainable parameters
    model.print_num_params() # Output will be around 6.6M for B1 with a replaced classifier

    # Create a dummy input tensor
    # Batch size = 4, Channels = 3, Height = 240, Width = 240
    dummy_input = torch.randn(4, 3, INPUT_SIZE[0], INPUT_SIZE[1])

    # Put model in evaluation mode for inference example (disables dropout, etc.)
    model.eval()

    # Perform a forward pass
    with torch.no_grad(): # Disable gradient calculation for inference
        logits1, logits2 = model(dummy_input)
        # Or just: logits = model(dummy_input) # If forward returns only logits

    # Print output shapes
    print(f"\nLogits 1 shape: {logits1.shape}") # Should be [batch_size, num_classes] e.g., [4, 10]
    # print(f"Logits 2 shape: {logits2.shape}") # If you return two

    # You can access the criterion and optimizer like this:
    print(f"Criterion: {model.criterion}")
    print(f"Optimizer: {model.optimizer}")

    # Example of accessing parameters for the optimizer (if you were training)
    # optimizer = model.optimizer
    # loss = model.criterion(logits1, target_labels) # Assuming you have target_labels
    # loss.backward()
    # optimizer.step()
# vgg16_custom.py

import warnings
from typing import Any, List, Union, Dict, cast, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

# Configuration dictionary for VGG architectures (from torchvision)
# Defines layers: numbers are output channels, 'M' is MaxPool2d
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # VGG16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], # VGG19
}

# Helper function to create VGG feature layers (from torchvision)
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGG16Custom(nn.Module):
    """
    Custom VGG16 implementation (with optional Batch Normalization)
    with a configurable number of classes, learning rate, and support
    for pretrained weights.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.0005, # Learning rate for the optimizer
        pretrained: bool = True, # Load pretrained ImageNet weights
        batch_norm: bool = True, # Use VGG16_bn variant
        dropout: float = 0.5 # Dropout rate for classifier
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout
        self.batch_norm = batch_norm # Store if BN is used

        # Select VGG configuration
        vgg_cfg = cfgs["D"] # 'D' corresponds to VGG16

        # --- Feature Extractor (Convolutional Layers) ---
        self.features = make_layers(vgg_cfg, batch_norm=self.batch_norm)

        # --- Adaptive Pooling ---
        # Ensures fixed output size before classifier (standard VGG is 7x7)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Classifier (Fully Connected Layers) ---
        # Input features: 512 channels * 7 * 7 spatial size = 25088
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            # --- Final layer adapted to num_classes ---
            nn.Linear(4096, self.num_classes),
        )

        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                print(f"Initializing ONLY the final classifier layer (classifier.6) for {self.num_classes} classes.")
                self._initialize_weights_for_specific_modules([self.classifier[6]]) # Index 6 is the last Linear layer
                print("Pretrained weights loaded and final layer initialized.")
            # Fallback initialization is handled within load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        # Using AdamW, adjust if needed (original VGG used SGD with momentum)
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    def load_pretrained_weights(self) -> bool:
        """
        Downloads pretrained VGG16 weights (BN or regular) and loads only
        compatible layers, explicitly filtering out the final classifier layer.
        """
        # Choose the correct URL based on batch_norm flag
        if self.batch_norm:
            model_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
            print("Attempting to load pretrained VGG16_bn weights...")
        else:
            model_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
            print("Attempting to load pretrained VGG16 weights...")

        try:
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            model_state_dict = self.state_dict()

            # Filter the pretrained state dict
            keys_to_load = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

            # Explicitly remove final classifier keys
            final_classifier_weight_key = 'classifier.6.weight'
            final_classifier_bias_key = 'classifier.6.bias'
            keys_to_load.pop(final_classifier_weight_key, None)
            keys_to_load.pop(final_classifier_bias_key, None)
            print(f"Removed '{final_classifier_weight_key}' and '{final_classifier_bias_key}' from keys to load.")
            print(f"Filtered state dict. Will attempt to load {len(keys_to_load)} key(s).")

            # Load the filtered state dict
            self.load_state_dict(keys_to_load, strict=False)
            print("Pretrained weights loaded successfully for compatible layers.")
            return True # Indicate success

        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}")
            print("Fallback: Proceeding with random initialization for ALL layers.")
            # Initialize the entire network if loading fails
            self._initialize_weights_for_specific_modules(self.modules())
            return False # Indicate failure

    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """
        Initializes weights for Conv2d, Linear, and BatchNorm2d layers.
        Uses Kaiming Normal for Conv2d (good with ReLU/BN).
        """
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                # Kaiming initialization is generally preferred with ReLU
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) # More similar to original VGG init for Linear
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def print_num_params(self):
        """Prints the total and trainable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    # Configuration
    class CONFIGURE_DUMMY:
        num_classes = 4       # Example: Your specific number of classes
        learning_rate = 0.0001 # Example LR for VGG fine-tuning
        batch_size = 16       # VGG can be memory intensive
        num_epochs = 1        # Minimal test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate VGG16_bn (Pretrained) ---
    print("\n--- Creating VGG16Custom (BN=True, Pretrained, 4 classes) ---")
    myModel_pretrained_bn = VGG16Custom(
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        batch_norm=True, # Use batch norm variant
        dropout=0.5
    ).to(device=device)
    myModel_pretrained_bn.print_num_params()

    # --- Instantiate VGG16 (No BN, Pretrained) ---
    print("\n--- Creating VGG16Custom (BN=False, Pretrained, 10 classes) ---")
    myModel_pretrained_nobn = VGG16Custom(
        num_classes=10, # Different class count
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        batch_norm=False, # Use original VGG variant
        dropout=0.5
    ).to(device=device)
    myModel_pretrained_nobn.print_num_params()

    # --- Instantiate VGG16_bn (From Scratch) ---
    print("\n--- Creating VGG16Custom (BN=True, From Scratch, 4 classes) ---")
    myModel_scratch_bn = VGG16Custom(
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False, # Train from scratch
        batch_norm=True,
        dropout=0.5
    ).to(device=device)
    myModel_scratch_bn.print_num_params()

    # --- Test forward pass with dummy data (using pretrained BN model) ---
    print("\n--- Testing Forward Pass (Pretrained BN Model) ---")
    # VGG typically uses 224x224 input
    dummy_input = torch.randn(CONFIGURE_DUMMY.batch_size, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_input.shape}")

    myModel_pretrained_bn.eval()
    with torch.no_grad():
        output = myModel_pretrained_bn(dummy_input)

    print(f"Output shape: {output.shape}") # Should be [batch_size, num_classes]
    assert output.shape == (CONFIGURE_DUMMY.batch_size, CONFIGURE_DUMMY.num_classes), "Output shape is incorrect!"
    print("Forward pass successful.")

    # --- Simple Training Loop Example ---
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    model_to_train = myModel_pretrained_bn
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer

    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(32)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        model_to_train.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_to_train(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 2 == 0: # Print less frequently
                 print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")

    print("\n--- Dummy Training Complete ---")
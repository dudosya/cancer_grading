# alexnet_CBAM_3.py

import warnings
from typing import Any, Iterator, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

# --- Import CBAM ---
from .attention_modules import CBAM 

# --- Renamed Class ---
class AlexNetCustom_CBAM3(nn.Module):
    """
    Custom AlexNet implementation with CBAM modules inserted after each
    convolutional layer's ReLU activation within the feature extractor.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.001,
        pretrained: bool = True,
        dropout: float = 0.5,
        # Add CBAM config arguments
        cbam_reduction_ratio: int = 16,
        cbam_spatial_kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout
        self.cbam_reduction_ratio = cbam_reduction_ratio # Store for use below
        self.cbam_spatial_kernel_size = cbam_spatial_kernel_size

        # --- Feature Extractor (with CBAM insertions) ---
        # We build this layer by layer to insert CBAM easily
        features_layers: List[nn.Module] = []
        in_channels = 3

        # Layer 1: Conv -> ReLU -> CBAM -> MaxPool
        out_channels_1 = 64
        features_layers.extend([
            nn.Conv2d(in_channels, out_channels_1, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            CBAM(out_channels_1, self.cbam_reduction_ratio, self.cbam_spatial_kernel_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        in_channels = out_channels_1

        # Layer 2: Conv -> ReLU -> CBAM -> MaxPool
        out_channels_2 = 192
        features_layers.extend([
            nn.Conv2d(in_channels, out_channels_2, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            CBAM(out_channels_2, self.cbam_reduction_ratio, self.cbam_spatial_kernel_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])
        in_channels = out_channels_2

        # Layer 3: Conv -> ReLU -> CBAM
        out_channels_3 = 384
        features_layers.extend([
            nn.Conv2d(in_channels, out_channels_3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels_3, self.cbam_reduction_ratio, self.cbam_spatial_kernel_size),
        ])
        in_channels = out_channels_3

        # Layer 4: Conv -> ReLU -> CBAM
        out_channels_4 = 256
        features_layers.extend([
            nn.Conv2d(in_channels, out_channels_4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels_4, self.cbam_reduction_ratio, self.cbam_spatial_kernel_size),
        ])
        in_channels = out_channels_4

        # Layer 5: Conv -> ReLU -> CBAM -> MaxPool
        out_channels_5 = 256
        features_layers.extend([
            nn.Conv2d(in_channels, out_channels_5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            CBAM(out_channels_5, self.cbam_reduction_ratio, self.cbam_spatial_kernel_size),
            nn.MaxPool2d(kernel_size=3, stride=2),
        ])

        self.features = nn.Sequential(*features_layers)
        print(f"Initialized AlexNet features with CBAM after each Conv block.")

        # --- Adaptive Pooling ---
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # --- Classifier (Fully Connected Layers) ---
        # Input features: 256 (from last CBAM) * 6 * 6 = 9216
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes), # Final layer adapted
        )

        # --- Collect CBAM modules for initialization ---
        self.cbam_modules_list = [m for m in self.features if isinstance(m, CBAM)]
        print(f"Found {len(self.cbam_modules_list)} CBAM modules in features.")


        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                # Initialize ONLY the new CBAM modules and the final classifier layer
                print(f"Initializing new layers (ALL CBAM modules and final classifier) for {self.num_classes} classes.")
                # Init final Linear layer
                self._initialize_weights_for_specific_modules([self.classifier[6]])
                # Init all CBAM modules
                for cbam_module in self.cbam_modules_list:
                    self._initialize_weights_for_specific_modules(cbam_module.modules())
                print("Pretrained AlexNet weights loaded. CBAM and final layer initialized.")
            # Fallback handled in load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    # --- load_pretrained_weights uses the robust version ---
    def load_pretrained_weights(self) -> bool: # Return True on success, False on failure
        """
        Downloads pretrained AlexNet weights and loads only compatible layers (Conv, Linear except final).
        Relies on strict=False to ignore interspersed CBAM modules.
        """
        model_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        try:
            print("Downloading pretrained weights for AlexNet base...")
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            model_state_dict = self.state_dict()

            # --- Filter based on matching keys, excluding final classifier ---
            # This attempts to load original Conv/ReLU/MaxPool layers and early classifier layers
            # It relies on strict=False to skip the CBAM layers present in our model but not checkpoint
            keys_to_load = {}
            for k, v in pretrained_state_dict.items():
                if k in model_state_dict:
                    # Skip final classifier
                    if k == 'classifier.6.weight' or k == 'classifier.6.bias':
                        continue
                    # Check shape (optional but good practice)
                    if v.shape == model_state_dict[k].shape:
                        keys_to_load[k] = v
                    else:
                        print(f"Warning: Shape mismatch for layer {k}. Checkpoint: {v.shape}, Model: {model_state_dict[k].shape}. Skipping.")
                # else: key not in our model (shouldn't happen with this structure)

            print(f"Filtered state dict based on matching keys/shapes. Will attempt to load {len(keys_to_load)} key(s).")
            if not keys_to_load: print("Warning: No matching keys found!")

            # Load the filtered state dict. strict=False ignores CBAM layers in our model.
            self.load_state_dict(keys_to_load, strict=False)
            print("Pretrained weights loaded successfully for potentially compatible AlexNet layers.")
            return True # Indicate success

        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}")
            print("Fallback: Proceeding with random initialization for ALL layers.")
            self._initialize_weights_for_specific_modules(self.modules())
            return False # Indicate failure

    # --- _initialize_weights_for_specific_modules remains the same ---
    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear, BatchNorm2d layers."""
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Handle potential BN in CBAM
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)


    # --- Forward pass uses the new features sequence ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x) # CBAM modules applied internally here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # --- print_num_params remains the same ---
    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}") # Will be significantly higher due to CBAMs
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.0005
        batch_size = 16 # Smaller batch for potentially higher memory use
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate AlexNet with CBAM Option 3 (Pretrained) ---
    print("\n--- Creating AlexNetCustom_CBAM3 (Pretrained, 4 classes) ---")
    myModel_pretrained = AlexNetCustom_CBAM3( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        dropout=0.5
        # Use default CBAM params
    ).to(device=device)
    myModel_pretrained.print_num_params() # Note significantly increased param count

    # --- Instantiate from Scratch ---
    print("\n--- Creating AlexNetCustom_CBAM3 (From Scratch, 10 classes) ---")
    myModel_scratch = AlexNetCustom_CBAM3( # Use new class name
        num_classes=10,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False,
        dropout=0.5,
        cbam_reduction_ratio=8 # Example different CBAM param
    ).to(device=device)
    myModel_scratch.print_num_params()

    # --- Test forward pass ---
    print("\n--- Testing Forward Pass (Pretrained Model) ---")
    dummy_input = torch.randn(CONFIGURE_DUMMY.batch_size, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_input.shape}")
    myModel_pretrained.eval()
    with torch.no_grad():
        output = myModel_pretrained(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (CONFIGURE_DUMMY.batch_size, CONFIGURE_DUMMY.num_classes), "Output shape is incorrect!"
    print("Forward pass successful.")

    # --- Simple Training Loop Example ---
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    model_to_train = myModel_pretrained
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(32)], # Fewer samples
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )
    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        model_to_train.train(); running_loss = 0.0; print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
            outputs = model_to_train(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
            if (i+1) % 2 == 0: print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
    print("\n--- Dummy Training Complete ---")
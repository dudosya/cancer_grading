# alexnet_CBAM_1.py

import warnings
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .attention_modules import CBAM_5_Parallel as CBAM

class AlexNetCustom_CBAM1(nn.Module): # Renamed class
    """
    Custom AlexNet implementation with a single CBAM module inserted after
    the feature extractor (before avgpool).
    API includes configurable classes, learning rate, and pretrained weights.
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

        # --- Feature Extractor (Convolutional Layers) ---
        # Identical to original AlexNetCustom
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # --- CBAM Module (Inserted after features) ---
        # Input channels = output channels of the last feature layer = 256
        self.cbam = CBAM(
            gate_channels=256,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size
        )
        print(f"Initialized CBAM module with {256} channels after features.")

        # --- Adaptive Pooling ---
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        # --- Classifier (Fully Connected Layers) ---
        # Identical to original AlexNetCustom
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes), # Final layer adapted
        )

        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                # Initialize ONLY the new CBAM module and the final classifier layer
                print(f"Initializing new layers (CBAM and final classifier) for {self.num_classes} classes.")
                self._initialize_weights_for_specific_modules(self.cbam.modules()) # Init CBAM internals
                self._initialize_weights_for_specific_modules([self.classifier[6]]) # Init last Linear layer
                print("Pretrained weights loaded. CBAM and final layer initialized.")
            # Fallback initialization handled in load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    # --- load_pretrained_weights uses the robust version ---
    def load_pretrained_weights(self) -> bool: # Return True on success, False on failure
        """
        Downloads pretrained AlexNet weights and loads only compatible layers,
        explicitly filtering out the final classifier layer. CBAM layers are ignored
        as they don't exist in the checkpoint.
        """
        model_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        try:
            print("Downloading pretrained weights for AlexNet base...")
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            model_state_dict = self.state_dict() # Get structure of our current model (including CBAM)

            # Filter the pretrained state dict - keep only keys matching our model
            keys_to_load = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}

            # Explicitly remove final classifier keys from the keys to load
            final_classifier_weight_key = 'classifier.6.weight'
            final_classifier_bias_key = 'classifier.6.bias'
            keys_to_load.pop(final_classifier_weight_key, None)
            keys_to_load.pop(final_classifier_bias_key, None)
            print(f"Removed final classifier keys. Will attempt to load {len(keys_to_load)} key(s) from pretrained dict.")

            # Load the filtered state dict. `strict=False` handles the CBAM keys which are
            # in our model but not in keys_to_load.
            self.load_state_dict(keys_to_load, strict=False)
            print("Pretrained weights loaded successfully for compatible AlexNet layers.")
            return True # Indicate success

        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}")
            print("Fallback: Proceeding with random initialization for ALL layers.")
            self._initialize_weights_for_specific_modules(self.modules())
            return False # Indicate failure

    # --- _initialize_weights_for_specific_modules remains the same ---
    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear layers found within the iterator."""
        # Also initialize BatchNorm if CBAM includes it (original CBAM doesn't, but good practice)
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): # Add init for BN layers within CBAM if any
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    # --- Updated forward pass ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.cbam(x) # Apply CBAM here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    # --- print_num_params remains the same ---
    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}") # Will be slightly higher due to CBAM
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.0005
        batch_size = 32
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate the Custom AlexNet with CBAM (Pretrained) ---
    print("\n--- Creating AlexNetCustom_CBAM1 (Pretrained, 4 classes) ---")
    myModel_pretrained = AlexNetCustom_CBAM1( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        dropout=0.5
        # Use default CBAM params: reduction_ratio=16, spatial_kernel_size=7
    ).to(device=device)
    myModel_pretrained.print_num_params() # Note the increased param count

    # --- Instantiate from Scratch ---
    print("\n--- Creating AlexNetCustom_CBAM1 (From Scratch, 10 classes) ---")
    myModel_scratch = AlexNetCustom_CBAM1( # Use new class name
        num_classes=10,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False,
        dropout=0.5,
        cbam_reduction_ratio=8 # Example: Use different CBAM param
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
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(64)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )
    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        model_to_train.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(); outputs = model_to_train(inputs)
            loss = criterion(outputs, labels); loss.backward(); optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 5 == 0: print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
    print("\n--- Dummy Training Complete ---")
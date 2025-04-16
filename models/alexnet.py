# alexnet.py

import warnings
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

class AlexNetCustom(nn.Module):
    # ... (__init__ signature, layer definitions) ...

    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.001,
        pretrained: bool = True,
        dropout: float = 0.5
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout

        # --- Feature Extractor ---
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Dropout(p=self.dropout), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout), nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, self.num_classes), # Final layer adapted
        )

        # --- Weight Initialization or Loading ---
        if pretrained:
            # Try loading pretrained weights
            load_success = self.load_pretrained_weights() # Returns True on success, False on failure

            if load_success:
                # If loading succeeded, initialize ONLY the final classifier layer
                print(f"Initializing ONLY the final classifier layer (classifier.6) for {self.num_classes} classes.")
                self._initialize_weights_for_specific_modules([self.classifier[6]]) # Index 6 is the last Linear layer
                print("Pretrained weights loaded and final layer initialized.")
            # If loading failed (load_success is False), the load_pretrained_weights method
            # already handled initializing ALL layers from scratch as a fallback.
            # So, no further action needed here in the failure case.

        else:
            # If not pretrained, initialize ALL weights from scratch
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    # --- Updated load_pretrained_weights ---
    def load_pretrained_weights(self) -> bool: # Return True on success, False on failure
        """Downloads and loads pretrained ImageNet weights, adapting the final layer."""
        model_url = "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)

            # Explicitly remove the final classifier weights
            state_dict.pop('classifier.6.weight', None)
            state_dict.pop('classifier.6.bias', None)
            print("Removed final classifier keys from pretrained state_dict.")

            # Load the modified state dict
            self.load_state_dict(state_dict, strict=False)
            print("Pretrained weights loaded successfully for compatible layers.")
            return True # Indicate success

        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}")
            print("Fallback: Proceeding with random initialization for ALL layers.")
            # Initialize the entire network if loading fails
            self._initialize_weights_for_specific_modules(self.modules())
            return False # Indicate failure

    # --- _initialize_weights_for_specific_modules remains the same ---
    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear layers found within the iterator."""
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    # --- forward and print_num_params remain the same ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x); x = self.avgpool(x)
        x = torch.flatten(x, 1); x = self.classifier(x)
        return x

    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage (remains the same) ---
if __name__ == '__main__':
    # ... (rest of the __main__ block is identical) ...
    # Configuration
    class CONFIGURE_DUMMY:
        num_classes = 4       # Example: Your specific number of classes
        learning_rate = 0.0005 # Example LR
        batch_size = 32
        num_epochs = 1        # Minimal test

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate the Custom AlexNet (Pretrained) ---
    print("\n--- Creating AlexNetCustom (Pretrained, 4 classes) ---")
    myModel_pretrained = AlexNetCustom(
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        dropout=0.5
    ).to(device=device)
    myModel_pretrained.print_num_params()

    # --- Instantiate from Scratch ---
    print("\n--- Creating AlexNetCustom (From Scratch, 10 classes) ---")
    myModel_scratch = AlexNetCustom(
        num_classes=10, # Different number of classes
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False, # Train from scratch
        dropout=0.5
    ).to(device=device)
    myModel_scratch.print_num_params()

    # --- Test forward pass with dummy data (using pretrained model) ---
    print("\n--- Testing Forward Pass (Pretrained Model) ---")
    dummy_input = torch.randn(CONFIGURE_DUMMY.batch_size, 3, 224, 224).to(device) # Use 224x224
    print(f"Input shape: {dummy_input.shape}")

    # Set model to evaluation mode for testing forward pass
    myModel_pretrained.eval()
    with torch.no_grad():
        output = myModel_pretrained(dummy_input)

    print(f"Output shape: {output.shape}") # Should be [batch_size, num_classes]
    assert output.shape == (CONFIGURE_DUMMY.batch_size, CONFIGURE_DUMMY.num_classes), "Output shape is incorrect!"
    print("Forward pass successful.")

    # --- Simple Training Loop Example ---
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    # Use the pretrained model instance for the dummy training
    model_to_train = myModel_pretrained
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer # Get optimizer from the model instance

    # Dummy data loader
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(64)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        model_to_train.train() # Set model to training mode
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
            if (i + 1) % 5 == 0: # Print less frequently
                 print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")

    print("\n--- Dummy Training Complete ---")
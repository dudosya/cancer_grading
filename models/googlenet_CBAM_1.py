# googlenet_CBAM_1.py

import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Dict, Iterator # Added Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .attention_modules import *


# --- GoogLeNet Base Building Blocks ---
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1), # Original bug replicated
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class InceptionAux(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        conv_block: Optional[Callable[..., nn.Module]] = None,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
# --- End Base Building Blocks ---


# --- Custom GoogLeNet Implementation with CBAM (Option 1) ---

class GoogLeNetCustom_CBAM1(nn.Module):
    """
    GoogLeNet architecture with a Convolutional Block Attention Module (CBAM)
    inserted before the final average pooling layer (Option 1).
    """
    __constants__ = ["aux_logits"]

    def __init__(
        self,
        num_classes: int,
        lr: float, # Learning rate added
        pretrained: bool = True, # Flag to control pretrained weights
        aux_logits: bool = True,
        init_weights: bool = True, # Default to standard init if not pretrained
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        cbam_reduction_ratio: int = 16, # CBAM reduction ratio
        cbam_kernel_size: int = 7       # CBAM spatial kernel size
    ) -> None:
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.aux_logits = aux_logits

        # Use local BasicConv2d, Inception, InceptionAux
        conv_block = BasicConv2d
        inception_block = Inception
        inception_aux_block = InceptionAux

        # --- Define Layers (mirroring torchvision structure) ---
        # (Layer definitions remain the same - omitted for brevity)
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)


        # --- CBAM Module (Option 1: Before final AvgPool) ---
        self.cbam_final = CBAM(
            gate_channels=1024,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_kernel_size
        )
        print(f"Initialized CBAM module with {1024} channels before final AvgPool.")

        # --- Auxiliary Classifiers ---
        if aux_logits:
            # Ensure aux classifiers are defined before potentially initializing them
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        # --- Final Classifier ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes) # FC layer defined here

        # --- Weight Initialization or Loading ---
        if pretrained:
            self.load_pretrained_weights()
            # Initialize ONLY the newly added layers AFTER loading pretrained base
            print(f"Initializing new layers (fc, cbam_final, aux classifiers if enabled)...")
            self._initialize_weights_for_specific_modules([self.fc, self.cbam_final])
            if self.aux_logits:
                # Check they exist before trying to initialize
                if self.aux1: self._initialize_weights_for_specific_modules([self.aux1])
                if self.aux2: self._initialize_weights_for_specific_modules([self.aux2])
            print("Pretrained weights loaded. New layers initialized.")

        elif init_weights:
            print("Initializing all weights from scratch.")
            # Initialize ALL modules in the entire network
            self._initialize_weights_for_specific_modules(self.modules()) # Pass iterator over all modules

        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)


    def load_pretrained_weights(self):
        """Downloads and loads pretrained ImageNet weights, adapting for CBAM and num_classes."""
        model_url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            keys_to_remove = [k for k in state_dict if k.startswith('aux1.') or k.startswith('aux2.')]
            for key in keys_to_remove:
                state_dict.pop(key, None)
            self.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. Proceeding with random initialization.")
            # If loading fails, make sure init_weights logic runs (handled by __init__)


    # *** CORRECTED INITIALIZATION FUNCTION ***
    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """
        Initializes weights for Conv2d, Linear, and BatchNorm2d layers found
        within the provided module iterator (e.g., self.modules(), self.cbam_final.modules()).
        This function ONLY handles the base layer types directly.
        """
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                # print(f"Initializing Conv2d: {m}") # Debug print
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                 # print(f"Initializing Linear: {m}") # Debug print
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                if m.bias is not None:
                     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print(f"Initializing BatchNorm2d: {m}") # Debug print
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            # No recursive call needed here! Iteration handles traversing submodules.


    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # (Forward pass logic remains the same as before - omitted for brevity)
        # Stem
        x = self.conv1(x); x = self.maxpool1(x); x = self.conv2(x); x = self.conv3(x); x = self.maxpool2(x)
        # Inception 3
        x = self.inception3a(x); x = self.inception3b(x); x = self.maxpool3(x)
        # Inception 4
        x = self.inception4a(x)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None and self.training: aux1 = self.aux1(x)
        x = self.inception4b(x); x = self.inception4c(x); x = self.inception4d(x)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None and self.training: aux2 = self.aux2(x)
        x = self.inception4e(x); x = self.maxpool4(x)
        # Inception 5
        x = self.inception5a(x); x = self.inception5b(x)
        # --- Apply CBAM (Option 1) ---
        x = self.cbam_final(x)
        # Classifier
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x, aux2, aux1


    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Optional[Tensor], aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        # (eager_outputs logic remains the same - omitted for brevity)
        if self.training and self.aux_logits:
            assert aux1 is not None and aux2 is not None, "Aux outputs required in training"
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x: Tensor) -> GoogLeNetOutputs | Tensor:
        # (forward logic remains the same - omitted for brevity)
        x, aux2, aux1 = self._forward(x)
        if torch.jit.is_scripting():
            if not (self.training and self.aux_logits):
                 warnings.warn("Scripted GoogleNet+CBAM returns tuple, aux might be None.", stacklevel=2)
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def print_num_params(self):
        # (print_num_params logic remains the same - omitted for brevity)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':
    # (Example Usage block remains the same as before - omitted for brevity,
    #  but make sure it's included in your actual file to run directly)

    # Configuration (replace with your actual config/args)
    class CONFIGURE_DUMMY:
        num_classes = 4       # Example: Your specific number of classes
        learning_rate = 0.001
        batch_size = 8        # Smaller batch for easier testing if needed
        num_epochs = 3        # Short training run
        wandb = False         # Disable wandb logging for this test

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate the Custom GoogLeNet with CBAM ---
    print("\n--- Creating GoogLeNetCustom_CBAM1 (Pretrained) ---")
    myModel = GoogLeNetCustom_CBAM1(
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,   # Load pretrained weights for GoogLeNet base
        aux_logits=True,   # Keep aux logits enabled (optional)
    ).to(device=device)

    # Print parameter count (will include CBAM parameters)
    myModel.print_num_params()

    # Create dummy data loaders
    print("\n--- Setting up Dummy DataLoaders ---")
    dummy_train_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(32)]
    dummy_test_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(16)]
    train_loader = torch.utils.data.DataLoader(dummy_train_data, batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dummy_test_data, batch_size=CONFIGURE_DUMMY.batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # --- Use a simplified training loop for direct testing ---
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    criterion = myModel.criterion # Use criterion from model
    optimizer = myModel.optimizer # Use optimizer from model

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        myModel.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = myModel(inputs) # Forward pass

            # Handle potential tuple output
            if isinstance(outputs, GoogLeNetOutputs):
                loss_main = criterion(outputs.logits, labels)
                loss_aux1 = criterion(outputs.aux_logits1, labels)
                loss_aux2 = criterion(outputs.aux_logits2, labels)
                loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
                logits_for_acc = outputs.logits # Use main logits for accuracy check
            else:
                loss = criterion(outputs, labels)
                logits_for_acc = outputs

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if (i + 1) % 4 == 0: # Print mini-batch loss
                print(f"Epoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Training Loss: {running_loss / len(train_loader):.4f}")

        # Simple evaluation step
        myModel.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = myModel(images)
                # Ensure we handle potential tuple output even in eval if scripting forces it
                logits_for_eval = outputs.logits if isinstance(outputs, GoogLeNetOutputs) else outputs
                _, predicted = torch.max(logits_for_eval.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1} Test Accuracy: {(100 * correct / total):.2f} %")

    print("\n--- Dummy Training Complete ---")

    # --- Example: Creating without pretrained weights ---
    print("\n--- Creating GoogLeNetCustom_CBAM1 (From Scratch) ---")
    model_scratch = GoogLeNetCustom_CBAM1(
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False,
        init_weights=True # Explicitly initialize all weights
    ).to(device=device)
    model_scratch.print_num_params()
    print("Model created from scratch successfully.")
# googlenet_CBAM_1_SpatialKernel.py

import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Dict, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .attention_modules import CBAM_1_SpatialKernel as CBAM

# --- GoogLeNet Base Building Blocks ---
# (GoogLeNetOutputs, BasicConv2d, Inception, InceptionAux - same as before, omitted for brevity)
# Make sure these are included in your file
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


# --- Custom GoogLeNet Implementation with CBAM (Option 1 - Spatial Kernel Mod) ---

# --- Updated Class Name ---
class GoogLeNetCustom_CBAM1_SpatialKernel(nn.Module):
    """
    GoogLeNet architecture with CBAM_1_SpatialKernel inserted before the final
    average pooling layer (Option 1), allowing configuration of the spatial kernel.
    """
    __constants__ = ["aux_logits"]

    def __init__(
        self,
        num_classes: int,
        lr: float,
        pretrained: bool = True,
        aux_logits: bool = True,
        init_weights: bool = True,
        dropout: float = 0.2,
        dropout_aux: float = 0.7,
        cbam_reduction_ratio: int = 16,
        # --- Added Argument ---
        cbam_spatial_kernel_size: int = 7 # Default to 7, but configurable
    ) -> None:
        super().__init__()

        # --- Check Spatial Kernel Size ---
        if cbam_spatial_kernel_size % 2 == 0:
             warnings.warn(f"cbam_spatial_kernel_size ({cbam_spatial_kernel_size}) is even. Padding might not work as expected for 'same' output size. Odd kernel size is recommended.")

        self.lr = lr
        self.num_classes = num_classes
        self.aux_logits = aux_logits

        conv_block = BasicConv2d
        inception_block = Inception
        inception_aux_block = InceptionAux

        # --- Define Layers ---
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


        # --- CBAM Module ---
        # Uses the imported CBAM (aliased from CBAM_1_SpatialKernel)
        # Passes the configurable spatial kernel size
        self.cbam_final = CBAM(
            gate_channels=1024,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size # Pass the argument
        )
        # The print statement is now inside CBAM_1_SpatialKernel

        # --- Auxiliary Classifiers ---
        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux)
        else:
            self.aux1 = None
            self.aux2 = None

        # --- Final Classifier ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes)

        # --- Weight Initialization or Loading ---
        if pretrained:
            self.load_pretrained_weights()
            print(f"Initializing new layers (fc, cbam_final, aux classifiers if enabled)...")
            # Initialize FC layer
            self._initialize_weights_for_specific_modules([self.fc])
            # Initialize the CBAM module by iterating its internal modules
            self._initialize_weights_for_specific_modules(self.cbam_final.modules())
            # Initialize Aux layers if they exist
            if self.aux_logits:
                if self.aux1: self._initialize_weights_for_specific_modules(self.aux1.modules())
                if self.aux2: self._initialize_weights_for_specific_modules(self.aux2.modules())
            print("Pretrained weights loaded. New layers initialized.")

        elif init_weights:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())

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


    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear, BatchNorm2d layers."""
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)


    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # (Forward pass logic remains the same as googlenet_CBAM_1.py)
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
        x = self.cbam_final(x) # This now uses CBAM_1_SpatialKernel
        # Classifier
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x, aux2, aux1


    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Optional[Tensor], aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        # (eager_outputs logic remains the same)
        if self.training and self.aux_logits:
            assert aux1 is not None and aux2 is not None, "Aux outputs required in training"
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x: Tensor) -> GoogLeNetOutputs | Tensor:
        # (forward logic remains the same)
        x, aux2, aux1 = self._forward(x)
        if torch.jit.is_scripting():
            if not (self.training and self.aux_logits):
                 warnings.warn("Scripted GoogleNet+CBAM1_SK returns tuple, aux might be None.", stacklevel=2)
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

    def print_num_params(self):
        # (print_num_params logic remains the same)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.001
        batch_size = 8
        num_epochs = 1 # Minimal test
        wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate with default spatial kernel (7) ---
    print("\n--- Creating GoogLeNetCustom_CBAM1_SpatialKernel (Pretrained, Spatial Kernel=7) ---")
    myModel_k7 = GoogLeNetCustom_CBAM1_SpatialKernel( # Use updated class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        aux_logits=True,
        cbam_spatial_kernel_size=7 # Explicitly pass 7 (default)
    ).to(device=device)
    myModel_k7.print_num_params()

    # --- Instantiate with different spatial kernel (3) ---
    print("\n--- Creating GoogLeNetCustom_CBAM1_SpatialKernel (Pretrained, Spatial Kernel=3) ---")
    myModel_k3 = GoogLeNetCustom_CBAM1_SpatialKernel( # Use updated class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        aux_logits=True,
        cbam_spatial_kernel_size=3 # Pass 3
    ).to(device=device)
    myModel_k3.print_num_params() # Parameter count might differ slightly

    # --- Instantiate with different spatial kernel (5) ---
    print("\n--- Creating GoogLeNetCustom_CBAM1_SpatialKernel (Pretrained, Spatial Kernel=5) ---")
    myModel_k5 = GoogLeNetCustom_CBAM1_SpatialKernel( # Use updated class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        aux_logits=True,
        cbam_spatial_kernel_size=5 # Pass 5
    ).to(device=device)
    myModel_k5.print_num_params() # Parameter count might differ slightly

    # --- Test one of the models (e.g., k=3) ---
    print("\n--- Testing Model with Spatial Kernel=3 ---")
    # (Using the simplified training loop from previous examples)
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(16)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(8)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=False
    )

    criterion = myModel_k3.criterion
    optimizer = myModel_k3.optimizer

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        myModel_k3.train()
        running_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = myModel_k3(inputs)
            if isinstance(outputs, GoogLeNetOutputs):
                loss = criterion(outputs.logits, labels) + 0.3 * criterion(outputs.aux_logits1, labels) + 0.3 * criterion(outputs.aux_logits2, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")

        myModel_k3.eval()
        correct = 0; total = 0
        with torch.no_grad():
             for images, labels in test_loader:
                 images, labels = images.to(device), labels.to(device)
                 outputs = myModel_k3(images)
                 logits = outputs.logits if isinstance(outputs, GoogLeNetOutputs) else outputs
                 _, predicted = torch.max(logits.data, 1)
                 total += labels.size(0)
                 correct += (predicted == labels).sum().item()
        if total > 0: print(f"  Test Accuracy: {(100 * correct / total):.2f} %")

    print("\n--- Dummy Testing Complete ---")
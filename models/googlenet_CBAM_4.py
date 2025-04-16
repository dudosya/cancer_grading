# googlenet_CBAM_4.py

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

from .attention_modules import CBAM_6_Residual as CBAM


# --- GoogLeNet Base Building Blocks (Inception Modified) ---

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

# --- Modified Inception Module (Option 4) ---
class Inception_With_CBAM(nn.Module):
    """
    Inception module with an integrated CBAM applied to the concatenated output.
    """
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
        # Add CBAM parameters to init
        cbam_reduction_ratio: int = 16,
        cbam_kernel_size: int = 7
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

        # --- CBAM Integration ---
        # Calculate total output channels for the CBAM module
        output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        # Instantiate CBAM internally
        self.cbam = CBAM(
            gate_channels=output_channels,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_kernel_size
        )
        # No need to store output_channels separately anymore

    def _forward(self, x: Tensor) -> List[Tensor]:
        """Calculates outputs of the four branches."""
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x: Tensor) -> Tensor:
        """Concatenates branch outputs and applies internal CBAM."""
        outputs = self._forward(x)
        x_cat = torch.cat(outputs, 1)
        # Apply CBAM to the concatenated output
        x_refined = self.cbam(x_cat)
        return x_refined

# --- InceptionAux remains the same ---
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


# --- Custom GoogLeNet Implementation with CBAM (Option 4) ---

class GoogLeNetCustom_CBAM4(nn.Module):
    """
    GoogLeNet architecture with CBAM integrated *inside* each Inception block
    (Option 4).
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
        cbam_kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.aux_logits = aux_logits

        conv_block = BasicConv2d
        # Use the modified Inception module that contains CBAM
        inception_block_with_cbam = partial(
            Inception_With_CBAM,
            cbam_reduction_ratio=cbam_reduction_ratio,
            cbam_kernel_size=cbam_kernel_size
        )
        inception_aux_block = InceptionAux

        # --- Define Layers using the modified Inception block ---
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # --- Stage 3 ---
        self.inception3a = inception_block_with_cbam(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block_with_cbam(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # --- Stage 4 ---
        self.inception4a = inception_block_with_cbam(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block_with_cbam(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block_with_cbam(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block_with_cbam(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block_with_cbam(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # --- Stage 5 ---
        self.inception5a = inception_block_with_cbam(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block_with_cbam(832, 384, 192, 384, 48, 128, 128)

        # No separate CBAM modules needed here

        # --- Auxiliary Classifiers ---
        if aux_logits:
            # Aux classifiers still branch off the output of the *Inception block*
            # (which now includes CBAM internally)
            self.aux1 = inception_aux_block(512, num_classes, dropout=dropout_aux) # Attached after 4a
            self.aux2 = inception_aux_block(528, num_classes, dropout=dropout_aux) # Attached after 4d
        else:
            self.aux1 = None
            self.aux2 = None

        # --- Final Classifier ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1024, num_classes) # Input from last Inception block (5b)

        # --- List of Inception modules containing CBAM ---
        # Needed for targeted initialization when using pretraining
        self.inception_modules_with_cbam = [
             self.inception3a, self.inception3b,
             self.inception4a, self.inception4b, self.inception4c, self.inception4d, self.inception4e,
             self.inception5a, self.inception5b
        ]
        print(f"Initialized {len(self.inception_modules_with_cbam)} Inception blocks with internal CBAM.")


        # --- Weight Initialization or Loading ---
        if pretrained:
            self.load_pretrained_weights()
            print(f"Initializing new layers (fc, internal CBAMs, aux classifiers if enabled)...")
            # Initialize FC layer
            self._initialize_weights_for_specific_modules([self.fc])
            # Initialize the CBAM submodule within each Inception block
            for inception_module in self.inception_modules_with_cbam:
                 # Pass the iterator for the internal .cbam module
                self._initialize_weights_for_specific_modules(inception_module.cbam.modules())
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
            # Remove FC and Aux weights
            state_dict.pop('fc.weight', None); state_dict.pop('fc.bias', None)
            keys_to_remove = [k for k in state_dict if k.startswith('aux1.') or k.startswith('aux2.')]
            for key in keys_to_remove: state_dict.pop(key, None)
            # Weights for the *internal* CBAM modules won't be in the state_dict.
            # We also need to remove the original Inception branch weights if their names clash?
            # No, strict=False handles keys in the model not present in state_dict.
            # It also handles keys in state_dict not present in model (we removed fc/aux).
            # The base branch weights (e.g., inception3a.branch1...) should load correctly.
            self.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"Error loading pretrained weights: {e}. Proceeding with random initialization.")


    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """
        Initializes weights for Conv2d, Linear, and BatchNorm2d layers found
        within the provided module iterator.
        """
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
        # Stem
        x = self.conv1(x); x = self.maxpool1(x); x = self.conv2(x); x = self.conv3(x); x = self.maxpool2(x)

        # --- Stage 3 ---
        # CBAM is now applied internally within these calls
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # --- Stage 4 ---
        x = self.inception4a(x)
        aux1_input = x # Input to Aux1 is the *output* of the modified inception4a (which includes CBAM)
        aux1: Optional[Tensor] = None
        if self.aux1 is not None and self.training: aux1 = self.aux1(aux1_input)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2_input = x # Input to Aux2 is the *output* of the modified inception4d (which includes CBAM)
        aux2: Optional[Tensor] = None
        if self.aux2 is not None and self.training: aux2 = self.aux2(aux2_input)

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # --- Stage 5 ---
        x = self.inception5a(x)
        x = self.inception5b(x) # Final output now includes internal CBAM

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

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
                 warnings.warn("Scripted GoogleNet+CBAM4 returns tuple, aux might be None.", stacklevel=2)
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
    # Configuration
    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.001
        batch_size = 2 # Small batch for testing
        num_epochs = 1 # Minimal epochs for quick test
        wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate the Model (Option 4) ---
    print("\n--- Creating GoogLeNetCustom_CBAM4 (Pretrained) ---")
    myModel = GoogLeNetCustom_CBAM4( # Use the new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        aux_logits=True,
    ).to(device=device)

    myModel.print_num_params() # Parameter count similar to Option 3

    # Dummy data loaders
    print("\n--- Setting up Dummy DataLoaders ---")
    dummy_train_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(8)]
    dummy_test_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(4)]
    train_loader = torch.utils.data.DataLoader(dummy_train_data, batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dummy_test_data, batch_size=CONFIGURE_DUMMY.batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Simplified training loop
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    criterion = myModel.criterion
    optimizer = myModel.optimizer

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        myModel.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = myModel(inputs)

            if isinstance(outputs, GoogLeNetOutputs):
                loss_main = criterion(outputs.logits, labels)
                loss_aux1 = criterion(outputs.aux_logits1, labels)
                loss_aux2 = criterion(outputs.aux_logits2, labels)
                loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2
            else:
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i + 1) % 2 == 0:
                 print(f"Epoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}], Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} Average Training Loss: {running_loss / len(train_loader):.4f}")

        # Simple evaluation step
        myModel.eval()
        correct = 0; total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = myModel(images)
                logits_for_eval = outputs.logits if isinstance(outputs, GoogLeNetOutputs) else outputs
                _, predicted = torch.max(logits_for_eval.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if total > 0:
            print(f"Epoch {epoch+1} Test Accuracy: {(100 * correct / total):.2f} %")
        else:
            print(f"Epoch {epoch+1} Test Accuracy: N/A (no samples)")


    print("\n--- Dummy Training Complete ---")

    # --- Example: Creating without pretrained weights ---
    print("\n--- Creating GoogLeNetCustom_CBAM4 (From Scratch) ---")
    model_scratch = GoogLeNetCustom_CBAM4( # Use the new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False,
        init_weights=True
    ).to(device=device)
    model_scratch.print_num_params()
    print("Model created from scratch successfully.")
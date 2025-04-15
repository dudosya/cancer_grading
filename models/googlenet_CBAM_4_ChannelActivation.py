# googlenet_CBAM_4_ChannelActivation.py

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

# --- Import the Modified CBAM ---
# Import CBAM_2_ChannelActivation
from .attention_modules import CBAM_2_ChannelActivation as CBAM

# --- GoogLeNet Base Building Blocks (Inception Modified for Channel Activation CBAM) ---

GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

class BasicConv2d(nn.Module):
    # (Identical to previous versions)
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x); x = self.bn(x); return F.relu(x, inplace=True)

# --- Modified Inception Module (Option 4 - Channel Activation) ---
# --- Renamed Class ---
class Inception_With_CBAM_CA(nn.Module): # CA for Channel Activation
    """
    Inception module with an integrated CBAM_2_ChannelActivation applied to the
    concatenated output, allowing configurable channel activation type.
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
        # Add CBAM parameters to init, using the activation type
        cbam_reduction_ratio: int = 16,
        cbam_channel_activation_type: str = "relu", # Add this
        cbam_spatial_kernel_size: int = 7          # Keep spatial kernel fixed or make configurable too
    ) -> None:
        super().__init__()
        if conv_block is None: conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential( conv_block(in_channels, ch3x3red, kernel_size=1), conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1))
        self.branch3 = nn.Sequential( conv_block(in_channels, ch5x5red, kernel_size=1), conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1))
        self.branch4 = nn.Sequential( nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True), conv_block(in_channels, pool_proj, kernel_size=1))

        # --- CBAM Integration ---
        output_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj
        # Instantiate the imported CBAM (CBAM_2_ChannelActivation) internally
        self.cbam = CBAM( # Uses the imported alias
            gate_channels=output_channels,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size, # Using fixed or passed value
            channel_activation_type=cbam_channel_activation_type # Pass the specific activation type
        )

    def _forward(self, x: Tensor) -> List[Tensor]:
        # (Identical to previous Inception _forward)
        branch1 = self.branch1(x); branch2 = self.branch2(x); branch3 = self.branch3(x); branch4 = self.branch4(x)
        return [branch1, branch2, branch3, branch4]

    def forward(self, x: Tensor) -> Tensor:
        # (Identical to previous Inception_With_CBAM forward)
        outputs = self._forward(x)
        x_cat = torch.cat(outputs, 1)
        x_refined = self.cbam(x_cat) # Apply internal CBAM
        return x_refined

# --- InceptionAux remains the same ---
class InceptionAux(nn.Module):
    # (Identical to previous versions)
    def __init__( self, in_channels: int, num_classes: int, conv_block: Optional[Callable[..., nn.Module]] = None, dropout: float = 0.7) -> None:
        super().__init__(); conv_block = conv_block or BasicConv2d; self.conv = conv_block(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(128*4*4, 1024); self.fc2 = nn.Linear(1024, num_classes); self.dropout = nn.Dropout(p=dropout)
    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, (4, 4)); x = self.conv(x); x = torch.flatten(x, 1); x = F.relu(self.fc1(x), inplace=True)
        x = self.dropout(x); x = self.fc2(x); return x
# --- End Base Building Blocks ---


# --- Custom GoogLeNet Implementation with CBAM (Option 4 - Channel Activation Mod) ---

# --- Updated Class Name ---
class GoogLeNetCustom_CBAM4_ChannelActivation(nn.Module):
    """
    GoogLeNet architecture with CBAM_2_ChannelActivation integrated inside each
    Inception block (Option 4), allowing configurable channel activation type.
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
        cbam_spatial_kernel_size: int = 7, # Keep spatial kernel consistent if desired
        # --- Updated Argument Name ---
        cbam_channel_activation_type: str = "relu" # Use the specific name
    ) -> None:
        super().__init__()

        self.lr = lr
        self.num_classes = num_classes
        self.aux_logits = aux_logits

        conv_block = BasicConv2d
        # Use the modified Inception module (Inception_With_CBAM_CA)
        inception_block_with_cbam_ca = partial(
            Inception_With_CBAM_CA, # Use the renamed Inception class
            cbam_reduction_ratio=cbam_reduction_ratio,
            cbam_spatial_kernel_size=cbam_spatial_kernel_size, # Pass spatial kernel size too
            cbam_channel_activation_type=cbam_channel_activation_type # Pass the activation type here
        )
        inception_aux_block = InceptionAux

        # --- Define Layers using the modified Inception block ---
        # (Layer definitions remain the same, just use inception_block_with_cbam_ca)
        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3); self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1); self.conv3 = conv_block(64, 192, kernel_size=3, padding=1); self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception3a = inception_block_with_cbam_ca(192, 64, 96, 128, 16, 32, 32); self.inception3b = inception_block_with_cbam_ca(256, 128, 128, 192, 32, 96, 64); self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.inception4a = inception_block_with_cbam_ca(480, 192, 96, 208, 16, 48, 64); self.inception4b = inception_block_with_cbam_ca(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block_with_cbam_ca(512, 128, 128, 256, 24, 64, 64); self.inception4d = inception_block_with_cbam_ca(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block_with_cbam_ca(528, 256, 160, 320, 32, 128, 128); self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.inception5a = inception_block_with_cbam_ca(832, 256, 160, 320, 32, 128, 128); self.inception5b = inception_block_with_cbam_ca(832, 384, 192, 384, 48, 128, 128)

        # --- Auxiliary Classifiers ---
        if aux_logits:
            aux1_in_channels = 512; aux2_in_channels = 528 # Original sizes
            self.aux1 = inception_aux_block(aux1_in_channels, num_classes, dropout=dropout_aux)
            self.aux2 = inception_aux_block(aux2_in_channels, num_classes, dropout=dropout_aux)
        else: self.aux1 = self.aux2 = None

        # --- Final Classifier ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)); self.dropout = nn.Dropout(p=dropout)
        fc_in_channels = 1024; self.fc = nn.Linear(fc_in_channels, num_classes)

        # --- List of Inception modules containing CBAM ---
        self.inception_modules_with_cbam = [
             self.inception3a, self.inception3b, self.inception4a, self.inception4b,
             self.inception4c, self.inception4d, self.inception4e, self.inception5a, self.inception5b
        ]
        print(f"Initialized {len(self.inception_modules_with_cbam)} Inception blocks with internal CBAM (Channel Activation Configurable).")


        # --- Weight Initialization or Loading ---
        if pretrained:
            self.load_pretrained_weights()
            print(f"Initializing new layers (fc, internal CBAMs, aux classifiers if enabled)...")
            self._initialize_weights_for_specific_modules([self.fc])
            for inception_module in self.inception_modules_with_cbam:
                self._initialize_weights_for_specific_modules(inception_module.cbam.modules())
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
        # (Identical to previous load_pretrained_weights)
        model_url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)
            state_dict.pop('fc.weight', None); state_dict.pop('fc.bias', None)
            keys_to_remove = [k for k in state_dict if k.startswith('aux1.') or k.startswith('aux2.')]
            for key in keys_to_remove: state_dict.pop(key, None)
            self.load_state_dict(state_dict, strict=False)
        except Exception as e: print(f"Error loading pretrained weights: {e}. Proceeding with random initialization.")


    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        # (Identical to previous versions)
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2);
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2);
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)


    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # (Forward pass structure identical to googlenet_CBAM_4.py)
        # Stem
        x = self.conv1(x); x = self.maxpool1(x); x = self.conv2(x); x = self.conv3(x); x = self.maxpool2(x)
        # Stage 3 (CBAM internal)
        x = self.inception3a(x); x = self.inception3b(x); x = self.maxpool3(x)
        # Stage 4 (CBAM internal)
        x = self.inception4a(x)
        aux1: Optional[Tensor] = None;
        if self.aux1 is not None and self.training: aux1 = self.aux1(x) # Aux uses output of Inception+CBAM
        x = self.inception4b(x); x = self.inception4c(x)
        x = self.inception4d(x)
        aux2: Optional[Tensor] = None;
        if self.aux2 is not None and self.training: aux2 = self.aux2(x) # Aux uses output of Inception+CBAM
        x = self.inception4e(x); x = self.maxpool4(x)
        # Stage 5 (CBAM internal)
        x = self.inception5a(x); x = self.inception5b(x)
        # Classifier
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.dropout(x); x = self.fc(x)
        return x, aux2, aux1


    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux2: Optional[Tensor], aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        # (Identical to previous versions)
        if self.training and self.aux_logits: assert aux1 is not None and aux2 is not None; return GoogLeNetOutputs(x, aux2, aux1)
        else: return x

    def forward(self, x: Tensor) -> GoogLeNetOutputs | Tensor:
        # (Identical to previous versions)
        x, aux2, aux1 = self._forward(x)
        if torch.jit.is_scripting():
            if not (self.training and self.aux_logits): warnings.warn("Scripted GoogleNet+CBAM4_CA returns tuple, aux might be None.", stacklevel=2)
            return GoogLeNetOutputs(x, aux2, aux1)
        else: return self.eager_outputs(x, aux2, aux1)

    def print_num_params(self):
        # (Identical to previous versions)
        total_params=sum(p.numel() for p in self.parameters()); trainable_params=sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}"); print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':
    # Configuration
    class CONFIGURE_DUMMY: num_classes=4; learning_rate=0.001; batch_size=2; num_epochs=1; wandb=False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using device: {device}")

    # --- Instantiate with default channel activation (ReLU) ---
    print("\n--- Creating GoogLeNetCustom_CBAM4_ChannelActivation (Pretrained, Channel Act=ReLU) ---")
    myModel_relu = GoogLeNetCustom_CBAM4_ChannelActivation( # Use updated class name
        num_classes=CONFIGURE_DUMMY.num_classes, lr=CONFIGURE_DUMMY.learning_rate, pretrained=True, aux_logits=True,
        cbam_channel_activation_type="relu" # Pass default activation
    ).to(device=device)
    myModel_relu.print_num_params()

    # --- Instantiate with different channel activation (GELU) ---
    print("\n--- Creating GoogLeNetCustom_CBAM4_ChannelActivation (Pretrained, Channel Act=GELU) ---")
    myModel_gelu = GoogLeNetCustom_CBAM4_ChannelActivation( # Use updated class name
        num_classes=CONFIGURE_DUMMY.num_classes, lr=CONFIGURE_DUMMY.learning_rate, pretrained=True, aux_logits=True,
        cbam_channel_activation_type="gelu" # Pass activation type 'gelu'
    ).to(device=device)
    myModel_gelu.print_num_params() # Parameter count should be identical

    # --- Test one model (k=3) ---
    print("\n--- Testing Model with Channel Activation=GELU ---")
    # (Using the simplified training loop from previous examples)
    train_loader = torch.utils.data.DataLoader( [(torch.randn(3, 224, 224), torch.randint(0, 4, (1,)).item()) for _ in range(8)], batch_size=2, shuffle=True)
    test_loader = torch.utils.data.DataLoader( [(torch.randn(3, 224, 224), torch.randint(0, 4, (1,)).item()) for _ in range(4)], batch_size=2, shuffle=False)
    criterion = myModel_gelu.criterion; optimizer = myModel_gelu.optimizer

    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        myModel_gelu.train(); running_loss = 0.0; print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
            outputs = myModel_gelu(inputs) # Use the gelu model
            if isinstance(outputs, GoogLeNetOutputs): loss = criterion(outputs.logits, labels) + 0.3 * criterion(outputs.aux_logits1, labels) + 0.3 * criterion(outputs.aux_logits2, labels)
            else: loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
        myModel_gelu.eval(); correct = 0; total = 0
        with torch.no_grad():
             for images, labels in test_loader:
                 images, labels = images.to(device), labels.to(device); outputs = myModel_gelu(images) # Use the gelu model
                 logits = outputs.logits if isinstance(outputs, GoogLeNetOutputs) else outputs
                 _, predicted = torch.max(logits.data, 1); total += labels.size(0); correct += (predicted == labels).sum().item()
        if total > 0: print(f"  Test Accuracy: {(100 * correct / total):.2f} %")
        else: print(f"  Test Accuracy: N/A (no samples)")
    print("\n--- Dummy Testing Complete ---")

    # --- Example: Creating without pretrained weights ---
    print("\n--- Creating GoogLeNetCustom_CBAM4_ChannelActivation (From Scratch, Channel Act=SiLU) ---")
    model_scratch_silu = GoogLeNetCustom_CBAM4_ChannelActivation(
        num_classes=CONFIGURE_DUMMY.num_classes, lr=CONFIGURE_DUMMY.learning_rate, pretrained=False, init_weights=True,
        cbam_channel_activation_type="silu" # Example: Non-default activation from scratch
    ).to(device=device)
    model_scratch_silu.print_num_params()
    print("Model created from scratch successfully.")
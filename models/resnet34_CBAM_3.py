# resnet34_CBAM_3.py

import warnings
from typing import Any, List, Optional, Type, Union, Dict, cast, Iterator, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .attention_modules import CBAM 

# --- BasicBlock Definition (from torchvision.models.resnet) ---
# (Identical to the one in resnet34_custom.py - omitted for brevity)
# Make sure this is included in your file
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d( in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation,)
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion: int = 1
    def __init__( self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None, groups: int = 1, base_width: int = 64, dilation: int = 1, norm_layer: Optional[Callable[..., nn.Module]] = None,) -> None:
        super().__init__(); norm_layer = norm_layer or nn.BatchNorm2d
        if groups != 1 or base_width != 64: raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1: raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride); self.bn1 = norm_layer(planes); self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes); self.bn2 = norm_layer(planes); self.downsample = downsample; self.stride = stride
        # Store output planes for CBAM convenience if needed later, though expansion works too
        self.output_planes = planes * self.expansion
    def forward(self, x: Tensor) -> Tensor:
        identity = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out); out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out); return out
# --- End BasicBlock ---


# --- ResNet34 Custom Implementation with CBAM (Option 3 Placement) ---

# --- Renamed Class ---
class ResNet34Custom_CBAM3(nn.Module):
    """
    Custom ResNet-34 implementation with a CBAM module inserted after
    *every* BasicBlock.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.001,
        pretrained: bool = True,
        # Add CBAM config arguments
        cbam_reduction_ratio: int = 16,
        cbam_spatial_kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        # Store CBAM args to pass them down in _make_layer
        self.cbam_reduction_ratio = cbam_reduction_ratio
        self.cbam_spatial_kernel_size = cbam_spatial_kernel_size

        # Standard ResNet parameters
        block = BasicBlock
        layers = [3, 4, 6, 3] # ResNet-34 config
        self._norm_layer = nn.BatchNorm2d
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        # --- Stem ---
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Residual Layers (using modified _make_layer) ---
        self.layer1 = self._make_layer(block, 64, layers[0], add_cbam=True)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, add_cbam=True)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, add_cbam=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, add_cbam=True)

        # --- Classifier Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                print(f"Initializing new layers (ALL CBAM modules and final classifier)...")
                # Initialize final classifier
                self._initialize_weights_for_specific_modules([self.fc])
                # Initialize all CBAM modules within layer1, layer2, layer3, layer4
                count_cbam_init = 0
                for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
                    for module in layer:
                        # CBAM modules are directly added in the sequence by _make_layer
                        if isinstance(module, CBAM):
                            self._initialize_weights_for_specific_modules(module.modules())
                            count_cbam_init += 1
                print(f"Initialized {count_cbam_init} CBAM module(s).")
                print("Pretrained ResNet weights loaded. New layers initialized.")
            # Fallback handled in load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    # --- Modified _make_layer Function ---
    def _make_layer(
        self,
        block: Type[BasicBlock],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        add_cbam: bool = False, # New argument
    ) -> nn.Sequential:
        """
        Helper function to create a residual layer (stage).
        MODIFIED: Adds a CBAM module after each block if add_cbam=True.
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        # First block (handles downsampling)
        current_block = block(
            self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
        )
        layers.append(current_block)
        self.inplanes = planes * block.expansion # Update inplanes *after* the first block is defined

        # --- Add CBAM after the first block ---
        if add_cbam:
            cbam_planes = planes * block.expansion # Output channels of the block
            layers.append(CBAM(
                gate_channels=cbam_planes,
                reduction_ratio=self.cbam_reduction_ratio,
                spatial_kernel_size=self.cbam_spatial_kernel_size
            ))
            # print(f"  Added CBAM after first block in layer (out_planes={cbam_planes})") # Optional debug

        # Subsequent blocks
        for _ in range(1, blocks):
            current_block = block(
                self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                dilation=self.dilation, norm_layer=norm_layer,
            )
            layers.append(current_block)
            # --- Add CBAM after subsequent blocks ---
            if add_cbam:
                cbam_planes = planes * block.expansion
                layers.append(CBAM(
                    gate_channels=cbam_planes,
                    reduction_ratio=self.cbam_reduction_ratio,
                    spatial_kernel_size=self.cbam_spatial_kernel_size
                ))
                # print(f"  Added CBAM after subsequent block in layer (out_planes={cbam_planes})") # Optional debug


        return nn.Sequential(*layers)


    def load_pretrained_weights(self) -> bool:
        """
        Downloads pretrained ResNet-34 weights. Attempts to load weights for Conv/BN
        layers by matching names, filtering out the final FC layer. Relies on
        strict=False to ignore CBAM layers and handle potential name mismatches
        due to interspersed layers.
        """
        # (Identical to ResNet34Custom load_pretrained_weights - robust version)
        model_url = "https://download.pytorch.org/models/resnet34-b627a593.pth"; print("Attempting to load ResNet-34 pretrained weights...")
        try:
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True); model_state_dict = self.state_dict()
            keys_to_load = {}
            pretrained_keys = list(pretrained_state_dict.keys()); model_keys = list(model_state_dict.keys())
            # Try to load based on matching names, excluding FC
            for key_pretrain in pretrained_keys:
                 if key_pretrain in model_keys:
                     # Skip final classifier
                     if key_pretrain == 'fc.weight' or key_pretrain == 'fc.bias': continue
                     # Check shape compatibility
                     if pretrained_state_dict[key_pretrain].shape == model_state_dict[key_pretrain].shape:
                         keys_to_load[key_pretrain] = pretrained_state_dict[key_pretrain]
                     else: print(f"Warning: Shape mismatch for layer {key_pretrain}. Skipping loading.")

            print(f"Filtered state dict based on matching keys/shapes. Will attempt to load {len(keys_to_load)} key(s).")
            if not keys_to_load: print("Warning: No matching keys found!")

            # Load the filtered state dict - strict=False ignores CBAM modules in our model
            # and might handle some layer name shifts if Conv/BN names still match.
            self.load_state_dict(keys_to_load, strict=False)
            print("Pretrained weights loaded successfully for potentially compatible ResNet layers.")
            return True # Success
        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}"); print("Fallback: Random initialization."); self._initialize_weights_for_specific_modules(self.modules()); return False


    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear, and BatchNorm2d layers."""
        # (Identical to ResNet34Custom initialization)
        for m in module_iterator:
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)


    # --- Forward pass uses layers which now include CBAM ---
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x) # Includes CBAM internally now
        x = self.layer2(x) # Includes CBAM internally now
        x = self.layer3(x) # Includes CBAM internally now
        x = self.layer4(x) # Includes CBAM internally now
        # No separate CBAM call needed here
        x = self.avgpool(x); x = torch.flatten(x, 1); x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def print_num_params(self):
        """Prints the total and trainable parameters in the model."""
        # (Identical to ResNet34Custom print_num_params)
        total_params = sum(p.numel() for p in self.parameters()); trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}"); print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.001
        batch_size = 16 # Smaller batch might be needed
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate ResNet34 with CBAM Option 3 (Pretrained) ---
    print("\n--- Creating ResNet34Custom_CBAM3 (Pretrained, 4 classes) ---")
    myModel_pretrained = ResNet34Custom_CBAM3( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True
        # Use default CBAM params
    ).to(device=device)
    myModel_pretrained.print_num_params() # Much higher param count

    # --- Instantiate ResNet34 with CBAM Option 3 (From Scratch) ---
    print("\n--- Creating ResNet34Custom_CBAM3 (From Scratch, 10 classes) ---")
    myModel_scratch = ResNet34Custom_CBAM3( # Use new class name
        num_classes=10,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False, # Train from scratch
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
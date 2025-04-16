# resnet34_CBAM_1.py

import warnings
from typing import Any, List, Optional, Type, Union, Dict, cast, Iterator, Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

# --- Import CBAM ---
from .attention_modules import CBAM_5_Parallel as CBAM

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
    def forward(self, x: Tensor) -> Tensor:
        identity = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out); out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None: identity = self.downsample(x)
        out += identity; out = self.relu(out); return out
# --- End BasicBlock ---


# --- ResNet34 Custom Implementation with CBAM (Option 1 Placement) ---

# --- Renamed Class ---
class ResNet34Custom_CBAM1(nn.Module):
    """
    Custom ResNet-34 implementation with a single CBAM module inserted after
    the final residual layer (layer4), before avgpool.
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

        # --- Residual Layers ---
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # Output channels: 512 * expansion = 512

        # --- CBAM Module (Inserted after layer4) ---
        self.cbam = CBAM(
            gate_channels=512 * block.expansion, # 512 for ResNet34
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size
        )
        print(f"Initialized CBAM module with {512 * block.expansion} channels after layer4.")

        # --- Classifier Head ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, self.num_classes)

        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                # Initialize ONLY the new CBAM module and the final FC layer
                print(f"Initializing new layers (CBAM and final classifier) for {self.num_classes} classes.")
                self._initialize_weights_for_specific_modules(self.cbam.modules()) # Init CBAM internals
                self._initialize_weights_for_specific_modules([self.fc])         # Init last Linear layer
                print("Pretrained ResNet weights loaded. CBAM and final layer initialized.")
            # Fallback handled in load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)

    def _make_layer(
        self, block: Type[BasicBlock], planes: int, blocks: int,
        stride: int = 1, dilate: bool = False,
    ) -> nn.Sequential:
        # (Identical to ResNet34Custom _make_layer - omitted for brevity)
        # Make sure this is included in your file
        norm_layer = self._norm_layer; downsample = None; previous_dilation = self.dilation
        if dilate: self.dilation *= stride; stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential( conv1x1(self.inplanes, planes * block.expansion, stride), norm_layer(planes * block.expansion),)
        layers = []; layers.append( block( self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append( block( self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer,))
        return nn.Sequential(*layers)


    def load_pretrained_weights(self) -> bool:
        """
        Downloads pretrained ResNet-34 weights and loads only compatible layers,
        explicitly filtering out the final fully connected layer. CBAM layers are
        ignored as they don't exist in the checkpoint.
        """
        # (Identical to ResNet34Custom load_pretrained_weights - omitted for brevity)
        # Make sure this is included in your file
        model_url = "https://download.pytorch.org/models/resnet34-b627a593.pth"; print("Attempting to load ResNet-34 pretrained weights...")
        try:
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True); model_state_dict = self.state_dict()
            keys_to_load = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
            final_fc_weight_key = 'fc.weight'; final_fc_bias_key = 'fc.bias'
            keys_to_load.pop(final_fc_weight_key, None); keys_to_load.pop(final_fc_bias_key, None)
            print(f"Removed '{final_fc_weight_key}' and '{final_fc_bias_key}'. Will load {len(keys_to_load)} key(s).")
            self.load_state_dict(keys_to_load, strict=False); print("Pretrained weights loaded successfully for compatible ResNet layers."); return True
        except Exception as e:
            print(f"Error loading/processing pretrained weights: {e}"); print("Fallback: Random initialization."); self._initialize_weights_for_specific_modules(self.modules()); return False


    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear, and BatchNorm2d layers."""
        # (Identical to ResNet34Custom initialization - slightly modified to init BN in CBAM too)
        for m in module_iterator:
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)): nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    # --- Updated forward pass ---
    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.cbam(x) # Apply CBAM here
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
        batch_size = 32
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate ResNet34 with CBAM (Pretrained) ---
    print("\n--- Creating ResNet34Custom_CBAM1 (Pretrained, 4 classes) ---")
    myModel_pretrained = ResNet34Custom_CBAM1( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True
        # Use default CBAM params
    ).to(device=device)
    myModel_pretrained.print_num_params() # Note slightly higher param count

    # --- Instantiate ResNet34 with CBAM (From Scratch) ---
    print("\n--- Creating ResNet34Custom_CBAM1 (From Scratch, 10 classes) ---")
    myModel_scratch = ResNet34Custom_CBAM1( # Use new class name
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
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(64)],
        batch_size=CONFIGURE_DUMMY.batch_size, shuffle=True
    )
    for epoch in range(CONFIGURE_DUMMY.num_epochs):
        model_to_train.train(); running_loss = 0.0; print(f"\nEpoch [{epoch+1}/{CONFIGURE_DUMMY.num_epochs}]")
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device); optimizer.zero_grad()
            outputs = model_to_train(inputs); loss = criterion(outputs, labels)
            loss.backward(); optimizer.step(); running_loss += loss.item()
            if (i+1) % 5 == 0: print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
    print("\n--- Dummy Training Complete ---")
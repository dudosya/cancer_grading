# vgg16_CBAM_3.py

import warnings
from typing import Any, List, Union, Dict, cast, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

from .attention_modules import CBAM 


# --- VGG Base Components (make_layers MODIFIED) ---
cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # VGG16
}

# --- Modified make_layers function ---
def make_layers_with_cbam(
    cfg: List[Union[str, int]],
    batch_norm: bool = False,
    # Add CBAM args
    add_cbam: bool = True,
    cbam_reduction_ratio: int = 16,
    cbam_spatial_kernel_size: int = 7
) -> nn.Sequential:
    """
    Builds VGG feature layers, inserting CBAM after each Conv-(BN)-ReLU block.
    """
    layers: List[nn.Module] = []
    in_channels = 3
    print(f"Building VGG layers {'with' if add_cbam else 'without'} CBAM insertions.")
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

            # --- Insert CBAM after Conv-(BN)-ReLU block ---
            if add_cbam:
                cbam_module = CBAM(
                    gate_channels=v, # Output channels of the preceding conv layer
                    reduction_ratio=cbam_reduction_ratio,
                    spatial_kernel_size=cbam_spatial_kernel_size
                )
                layers += [cbam_module]
                # Optional: Print where CBAM was added
                # print(f"  Added CBAM after Conv block outputting {v} channels.")
            # --- End CBAM Insertion ---

            in_channels = v # Update in_channels for the next Conv layer
    return nn.Sequential(*layers)
# --- End VGG Base Components ---


# --- Custom VGG16 Implementation with CBAM (Option 3 Placement) ---

# --- Renamed Class ---
class VGG16Custom_CBAM3(nn.Module):
    """
    Custom VGG16 implementation (with optional BN) using CBAM inserted
    after *every* Conv-(BN)-ReLU block within the feature extractor.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.0005,
        pretrained: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.5,
        # Add CBAM config arguments
        cbam_reduction_ratio: int = 16,
        cbam_spatial_kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout
        self.batch_norm = batch_norm

        vgg_cfg = cfgs["D"] # VGG16 config

        # --- Feature Extractor (Built with modified make_layers) ---
        self.features = make_layers_with_cbam( # Use the modified function
            vgg_cfg,
            batch_norm=self.batch_norm,
            add_cbam=True, # Ensure CBAM is added
            cbam_reduction_ratio=cbam_reduction_ratio,
            cbam_spatial_kernel_size=cbam_spatial_kernel_size
        )

        # --- Adaptive Pooling ---
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Classifier (Fully Connected Layers) ---
        # Input features still 512 * 7 * 7 (CBAM doesn't change feature map size)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout),
            nn.Linear(4096, self.num_classes),
        )

        # --- Weight Initialization or Loading ---
        if pretrained:
            load_success = self.load_pretrained_weights()
            if load_success:
                print(f"Initializing new layers (ALL CBAM modules and final classifier)...")
                # Initialize final classifier
                self._initialize_weights_for_specific_modules([self.classifier[6]])
                # Initialize all CBAM modules within features
                count_cbam_init = 0
                for module in self.features:
                    if isinstance(module, CBAM):
                        self._initialize_weights_for_specific_modules(module.modules())
                        count_cbam_init += 1
                print(f"Initialized {count_cbam_init} CBAM module(s).")
                print("Pretrained VGG weights loaded. New layers initialized.")
            # Fallback handled in load_pretrained_weights
        else:
            print("Initializing all weights from scratch.")
            self._initialize_weights_for_specific_modules(self.modules())


        # --- Define Optimizer and Criterion ---
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)


    def load_pretrained_weights(self) -> bool:
        """
        Downloads pretrained VGG16 weights (BN or regular) and loads only
        compatible layers (Conv, BN, Linear except final). CBAM layers are ignored.
        """
        if self.batch_norm: model_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        else: model_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        print(f"Attempting to load {'VGG16_bn' if self.batch_norm else 'VGG16'} pretrained weights...")

        try:
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            model_state_dict = self.state_dict()

            # Filter pretrained dict to keep only keys matching our model structure
            # This automatically handles the fact that our layer indices will differ
            # from the original VGG due to the inserted CBAM modules, BUT the weights
            # for layers like features.0.weight, features.1.weight (Conv, BN)
            # will have matching *names* up until the first CBAM insertion.
            # We need a more careful matching.

            # --- More Careful Filtering for Interspersed Layers ---
            keys_to_load = {}
            pretrained_keys = list(pretrained_state_dict.keys())
            model_keys = list(model_state_dict.keys())

            # Iterate through layers based on the *original* structure to load matching weights
            # This assumes layer names like 'features.0.*', 'features.3.*' etc. match
            # between the original VGG checkpoint and our Conv/BN layers.
            # It relies on `strict=False` to ignore the CBAM layers in our model.
            for key_pretrain in pretrained_keys:
                if key_pretrain in model_keys:
                    # Check if it's the final classifier layer
                    if key_pretrain == 'classifier.6.weight' or key_pretrain == 'classifier.6.bias':
                        continue # Skip final classifier

                    # Check shape compatibility (optional but recommended)
                    if pretrained_state_dict[key_pretrain].shape == model_state_dict[key_pretrain].shape:
                        keys_to_load[key_pretrain] = pretrained_state_dict[key_pretrain]
                    else:
                        print(f"Warning: Shape mismatch for layer {key_pretrain}. Skipping loading.")
                # else: Key from pretrained dict not found in our model (e.g., if we removed layers)

            print(f"Filtered state dict based on matching keys/shapes. Will attempt to load {len(keys_to_load)} key(s).")
            if not keys_to_load:
                 print("Warning: No matching keys found to load pretrained weights!")


            # Load the filtered state dict - strict=False ignores CBAM layers in our model
            self.load_state_dict(keys_to_load, strict=False)
            print("Pretrained weights loaded successfully for compatible VGG layers.")
            return True # Success

        except Exception as e:
            print(f"Error loading or processing pretrained weights: {e}")
            print("Fallback: Proceeding with random initialization for ALL layers.")
            self._initialize_weights_for_specific_modules(self.modules())
            return False # Failure

    def _initialize_weights_for_specific_modules(self, module_iterator: Iterator[nn.Module]) -> None:
        """Initializes weights for Conv2d, Linear, and BatchNorm2d layers."""
        # (Identical to VGG16Custom initialization)
        for m in module_iterator:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    # --- Forward pass uses the features sequence which now includes CBAM ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x) # CBAM modules applied internally here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def print_num_params(self):
        """Prints the total and trainable parameters in the model."""
        # (Identical to VGG16Custom print_num_params)
        total_params = sum(p.numel() for p in self.parameters()); trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}"); print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.0001
        batch_size = 8 # Even smaller batch might be needed due to many CBAMs
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate VGG16_bn with CBAM Option 3 (Pretrained) ---
    print("\n--- Creating VGG16Custom_CBAM3 (BN=True, Pretrained, 4 classes) ---")
    myModel_pretrained_bn = VGG16Custom_CBAM3( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        batch_norm=True,
        dropout=0.5
        # Uses default CBAM reduction ratio and spatial kernel size
    ).to(device=device)
    myModel_pretrained_bn.print_num_params() # Significantly higher param count

    # --- Instantiate VGG16_bn with CBAM Option 3 (From Scratch) ---
    print("\n--- Creating VGG16Custom_CBAM3 (BN=True, From Scratch, 10 classes) ---")
    myModel_scratch_bn = VGG16Custom_CBAM3( # Use new class name
        num_classes=10,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=False, # Train from scratch
        batch_norm=True,
        dropout=0.5,
        cbam_reduction_ratio=8 # Example different CBAM param
    ).to(device=device)
    myModel_scratch_bn.print_num_params()

    # --- Test forward pass ---
    print("\n--- Testing Forward Pass (Pretrained BN Model) ---")
    dummy_input = torch.randn(CONFIGURE_DUMMY.batch_size, 3, 224, 224).to(device)
    print(f"Input shape: {dummy_input.shape}")
    myModel_pretrained_bn.eval()
    with torch.no_grad():
        output = myModel_pretrained_bn(dummy_input)
    print(f"Output shape: {output.shape}")
    assert output.shape == (CONFIGURE_DUMMY.batch_size, CONFIGURE_DUMMY.num_classes), "Output shape incorrect!"
    print("Forward pass successful.")

    # --- Simple Training Loop Example ---
    print(f"\n--- Starting Dummy Training for {CONFIGURE_DUMMY.num_epochs} epochs ---")
    model_to_train = myModel_pretrained_bn
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(16)], # Fewer samples
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
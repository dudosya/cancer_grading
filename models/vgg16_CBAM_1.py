# vgg16_CBAM_1.py

import warnings
from typing import Any, List, Union, Dict, cast, Iterator

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

# --- Import the Modified CBAM ---
# Import CBAM_5_Parallel (Parallel Application Logic)
from .attention_modules import CBAM_5_Parallel as CBAM

# --- VGG Base Components ---
# (cfgs dictionary and make_layers function - same as before, omitted for brevity)
# Make sure these are included in your file
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"], # VGG16
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"], # VGG19
}

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm: layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else: layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
# --- End VGG Base Components ---


# --- Custom VGG16 Implementation with Parallel CBAM (Option 1 Placement) ---

# --- Updated Class Name ---
class VGG16Custom_CBAM1_Parallel(nn.Module):
    """
    Custom VGG16 implementation (with optional BN) using CBAM_5_Parallel
    inserted after the feature extractor (Option 1 placement).
    API includes configurable classes, learning rate, and pretrained weights.
    """
    def __init__(
        self,
        num_classes: int = 1000,
        lr: float = 0.0005,
        pretrained: bool = True,
        batch_norm: bool = True, # Use VGG16_bn variant by default
        dropout: float = 0.5,
        # Add CBAM config arguments (used by CBAM_5_Parallel)
        cbam_reduction_ratio: int = 16,
        cbam_spatial_kernel_size: int = 7
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout
        self.batch_norm = batch_norm

        vgg_cfg = cfgs["D"] # VGG16 config

        # --- Feature Extractor (Convolutional Layers) ---
        self.features = make_layers(vgg_cfg, batch_norm=self.batch_norm)

        # --- CBAM Module (Inserted after features) ---
        # Input channels = output channels of the last feature layer = 512
        # Uses the imported CBAM alias (CBAM_5_Parallel)
        self.cbam = CBAM(
            gate_channels=512,
            reduction_ratio=cbam_reduction_ratio,
            spatial_kernel_size=cbam_spatial_kernel_size
        )
        print(f"Initialized CBAM (Parallel Logic) module with {512} channels after features.")

        # --- Adaptive Pooling ---
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # --- Classifier (Fully Connected Layers) ---
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Dropout(p=self.dropout),
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
                print("Pretrained VGG weights loaded. CBAM and final layer initialized.")
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
        compatible layers, filtering out the final classifier. CBAM layers are
        ignored as they don't exist in the checkpoint.
        """
        if self.batch_norm: model_url = "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth"
        else: model_url = "https://download.pytorch.org/models/vgg16-397923af.pth"
        print(f"Attempting to load {'VGG16_bn' if self.batch_norm else 'VGG16'} pretrained weights...")

        try:
            pretrained_state_dict = load_state_dict_from_url(model_url, progress=True)
            model_state_dict = self.state_dict()

            keys_to_load = {k: v for k, v in pretrained_state_dict.items() if k in model_state_dict}
            final_cls_w = 'classifier.6.weight'; final_cls_b = 'classifier.6.bias'
            keys_to_load.pop(final_cls_w, None); keys_to_load.pop(final_cls_b, None)
            print(f"Removed final classifier keys. Will attempt to load {len(keys_to_load)} key(s).")

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

    # --- Updated forward pass ---
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.cbam(x) # Apply Parallel CBAM here
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def print_num_params(self):
        """Prints the total and trainable parameters in the model."""
        # (Identical to VGG16Custom print_num_params)
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}") # Will be slightly higher due to CBAM
        print(f"Trainable Parameters: {trainable_params:,}")


# --- Example Usage ---
if __name__ == '__main__':

    class CONFIGURE_DUMMY:
        num_classes = 4
        learning_rate = 0.0001
        batch_size = 16
        num_epochs = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Instantiate VGG16_bn with Parallel CBAM (Pretrained) ---
    print("\n--- Creating VGG16Custom_CBAM1_Parallel (BN=True, Pretrained, 4 classes) ---")
    myModel_pretrained_bn = VGG16Custom_CBAM1_Parallel( # Use new class name
        num_classes=CONFIGURE_DUMMY.num_classes,
        lr=CONFIGURE_DUMMY.learning_rate,
        pretrained=True,
        batch_norm=True,
        dropout=0.5
        # Uses default CBAM reduction ratio and spatial kernel size
    ).to(device=device)
    myModel_pretrained_bn.print_num_params() # Note increased param count

    # --- Instantiate VGG16_bn with Parallel CBAM (From Scratch) ---
    print("\n--- Creating VGG16Custom_CBAM1_Parallel (BN=True, From Scratch, 10 classes) ---")
    myModel_scratch_bn = VGG16Custom_CBAM1_Parallel( # Use new class name
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
    model_to_train = myModel_pretrained_bn # Use the BN pretrained instance
    criterion = model_to_train.criterion
    optimizer = model_to_train.optimizer
    train_loader = torch.utils.data.DataLoader(
        [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE_DUMMY.num_classes, (1,)).item()) for _ in range(32)],
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
            if (i+1) % 2 == 0: print(f"  Batch [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        print(f"  Avg Train Loss: {running_loss / len(train_loader):.4f}")
    print("\n--- Dummy Training Complete ---")
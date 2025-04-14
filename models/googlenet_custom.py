# googlenet_custom.py

import warnings
from collections import namedtuple
from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.hub import load_state_dict_from_url

# --- Define Output Structure ---
GoogLeNetOutputs = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
GoogLeNetOutputs.__annotations__ = {"logits": Tensor, "aux_logits2": Optional[Tensor], "aux_logits1": Optional[Tensor]}

# --- Building Blocks (Copied/adapted from torchvision.models.googlenet) ---

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
            # Note: Pytorch's torchvision implementation uses kernel_size=3 here due to a historical bug.
            # We replicate this behavior to ensure compatibility with pretrained weights.
            # See https://github.com/pytorch/vision/issues/906
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1),
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
    # Note: If loading pretrained weights, the original aux weights are generally not useful
    # as they weren't trained effectively or require fine-tuning.
    # We keep the structure but might need to re-initialize if used.
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

        self.fc1 = nn.Linear(128 * 4 * 4, 1024) # 2048 -> 128 * 4 * 4 based on adaptive_avg_pool2d output
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: Tensor) -> Tensor:
        # Input shape assumptions: aux1: N x 512 x 14 x 14, aux2: N x 528 x 14 x 14
        # Adaptive avg pool converts feature map to 4x4
        x = F.adaptive_avg_pool2d(x, (4, 4))
        # Shape becomes N x C x 4 x 4 (e.g., N x 512 x 4 x 4 or N x 528 x 4 x 4)

        x = self.conv(x)
        # Shape becomes N x 128 x 4 x 4

        x = torch.flatten(x, 1)
        # Shape becomes N x (128*4*4) = N x 2048

        x = F.relu(self.fc1(x), inplace=True)
        # Shape becomes N x 1024

        x = self.dropout(x)
        # Shape remains N x 1024

        x = self.fc2(x)
        # Shape becomes N x num_classes

        return x

# --- Custom GoogLeNet Implementation ---

class GoogLeNetCustom(nn.Module):
    # We don't include transform_input here, assuming normalization happens in DataLoader
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
    ) -> None:
        super().__init__()

        self.lr = lr # Store learning rate
        self.num_classes = num_classes # Store num_classes

        # Use our local BasicConv2d, Inception, InceptionAux
        conv_block = BasicConv2d
        inception_block = Inception
        inception_aux_block = InceptionAux

        self.aux_logits = aux_logits

        # --- Define Layers (mirroring torchvision structure) ---
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
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True) # Note: Original uses kernel_size=2 here, different from other maxpools

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

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
        # *** Adapt the final layer ***
        self.fc = nn.Linear(1024, num_classes) # Original output is 1024

        # --- Weight Initialization or Loading ---
        if pretrained:
            self.load_pretrained_weights()
            # Optional: re-initialize aux classifiers even if loading pretrained base
            # Since torchvision weights aux are not pretrained effectively
            if self.aux_logits:
                self._initialize_weights(modules_list=[self.aux1, self.aux2])
                print("Pretrained weights loaded for base model. Auxiliary classifiers re-initialized.")
            else:
                 print("Pretrained weights loaded for base model.")
            # Final FC layer is already handled in load_pretrained_weights

        elif init_weights:
            print("Initializing weights from scratch.")
            self._initialize_weights(modules_list=self.modules())


        # --- Define Optimizer and Criterion (as requested in API) ---
        # Common choices, can be customized further if needed
        self.criterion = nn.CrossEntropyLoss()
        # Using AdamW which is often preferred
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)


    def load_pretrained_weights(self):
        """Downloads and loads pretrained ImageNet weights, adapting the final layer."""
        model_url = "https://download.pytorch.org/models/googlenet-1378be20.pth"
        try:
            state_dict = load_state_dict_from_url(model_url, progress=True)

            # --- Adapt state dict for our model ---
            # 1. Remove weights for the final fully connected layer
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)

            # 2. Remove weights for auxiliary classifiers (as their num_classes differs)
            #    and they are often retrained anyway.
            keys_to_remove = [k for k in state_dict if k.startswith('aux1.') or k.startswith('aux2.')]
            for key in keys_to_remove:
                state_dict.pop(key, None)

            # --- Load the adapted state dict ---
            # `strict=False` ignores keys that are not found (our new fc, aux layers)
            self.load_state_dict(state_dict, strict=False)

            # --- Initialize the new layers that didn't get weights ---
            print(f"Initializing new layers (fc, aux classifiers if enabled) for {self.num_classes} classes.")
            self._initialize_weights(modules_list=[self.fc])
            if self.aux_logits:
                 self._initialize_weights(modules_list=[self.aux1, self.aux2])


        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Proceeding with random initialization.")
            self._initialize_weights(modules_list=self.modules())

    def _initialize_weights(self, modules_list) -> None:
        """Initializes weights using truncated normal for Conv/Linear and constants for BN."""
        for m in modules_list:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # Using Kaiming Normal is a common modern alternative
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Replicating original GoogLeNet init:
                torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        # Input: N x 3 x H x W (assuming H, W >= 224)

        # --- Stem ---
        # N x 3 x H x W -> N x 64 x H/2 x W/2 (approx)
        x = self.conv1(x)
        # N x 64 x H/2 x W/2 -> N x 64 x H/4 x W/4 (approx)
        x = self.maxpool1(x)
        # N x 64 x H/4 x W/4
        x = self.conv2(x)
        # N x 64 x H/4 x W/4 -> N x 192 x H/4 x W/4
        x = self.conv3(x)
        # N x 192 x H/4 x W/4 -> N x 192 x H/8 x W/8 (approx, e.g., 28x28 for 224 input)
        x = self.maxpool2(x)

        # --- Inception blocks 3 ---
        # N x 192 x 28 x 28 -> N x 256 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28 -> N x 480 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28 -> N x 480 x 14 x 14 (approx)
        x = self.maxpool3(x)

        # --- Inception blocks 4 ---
        # N x 480 x 14 x 14 -> N x 512 x 14 x 14
        x = self.inception4a(x)

        # --- Aux1 Output (if enabled and training) ---
        aux1: Optional[Tensor] = None
        if self.aux1 is not None and self.training:
            aux1 = self.aux1(x) # Connect aux1 after inception4a

        # N x 512 x 14 x 14 -> N x 512 x 14 x 14
        x = self.inception4b(x)
        # N x 512 x 14 x 14 -> N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14 -> N x 528 x 14 x 14
        x = self.inception4d(x)

        # --- Aux2 Output (if enabled and training) ---
        aux2: Optional[Tensor] = None
        if self.aux2 is not None and self.training:
            aux2 = self.aux2(x) # Connect aux2 after inception4d

        # N x 528 x 14 x 14 -> N x 832 x 14 x 14
        x = self.inception4e(x)
        # N x 832 x 14 x 14 -> N x 832 x 7 x 7 (approx)
        x = self.maxpool4(x)

        # --- Inception blocks 5 ---
        # N x 832 x 7 x 7 -> N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7 -> N x 1024 x 7 x 7
        x = self.inception5b(x)

        # --- Classifier ---
        # N x 1024 x 7 x 7 -> N x 1024 x 1 x 1
        x = self.avgpool(x)
        # N x 1024 x 1 x 1 -> N x 1024
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        # N x 1024 -> N x num_classes
        x = self.fc(x)

        # --- Return main logits and aux logits (if applicable) ---
        return x, aux2, aux1

    # This logic handles the different return types for training/eval and scripting
    @torch.jit.unused # Indicate this part might not be scriptable as is
    def eager_outputs(self, x: Tensor, aux2: Optional[Tensor], aux1: Optional[Tensor]) -> GoogLeNetOutputs | Tensor:
        # If training and aux logits are enabled, return the named tuple
        if self.training and self.aux_logits:
            # Ensure aux outputs are not None during training if aux_logits is True
            assert aux1 is not None and aux2 is not None, "Auxiliary outputs must be present during training with aux_logits=True"
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            # Otherwise, return only the main logits tensor
            return x

    def forward(self, x: Tensor) -> GoogLeNetOutputs | Tensor:
        # We removed _transform_input, assuming input is already normalized
        x, aux2, aux1 = self._forward(x)

        # Check if we are in scripting mode (less flexible return types)
        if torch.jit.is_scripting():
            # Scripting expects a consistent return type. Always return the tuple.
            # If not training or aux_logits is False, aux outputs will be None.
            if not (self.training and self.aux_logits):
                 warnings.warn("Scripted GoogLeNet always returns the GoogLeNetOutputs Tuple, but aux outputs might be None in eval mode or if aux_logits=False.", stacklevel=2)
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            # Use the eager_outputs logic for standard Python execution
            return self.eager_outputs(x, aux2, aux1)

    def print_num_params(self):
        """Prints the total and trainable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

# --- Example Usage (similar to your main.py snippet) ---
if __name__ == '__main__':

    # Configuration (replace with your actual config)
    class CONFIGURE:
        num_classes = 4 # Your specific number of classes
        learning_rate = 0.001
        batch_size = 32 # Example batch size
        num_epochs = 10 # Example epochs

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Instantiate the Custom GoogLeNet ---
    print("--- Creating GoogLeNetCustom (Pretrained) ---")
    myModel = GoogLeNetCustom(
        num_classes=CONFIGURE.num_classes,
        lr=CONFIGURE.learning_rate,
        pretrained=True, # Load pretrained weights
        aux_logits=True   # Enable auxiliary outputs (optional)
    ).to(device=device)

    # Create dummy data loaders (replace with your actual data)
    print("\n--- Setting up Dummy DataLoaders ---")
    # Create dummy datasets for demonstration
    dummy_train_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE.num_classes, (1,)).item()) for _ in range(128)]
    dummy_test_data = [(torch.randn(3, 224, 224), torch.randint(0, CONFIGURE.num_classes, (1,)).item()) for _ in range(64)]

    train_dataset = dummy_train_data
    test_dataset = dummy_test_data

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=CONFIGURE.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=CONFIGURE.batch_size, shuffle=False)
    print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # --- Use your Trainer (assuming it exists) ---
    # Replace 'trainer' with your actual trainer module import
    # from your_project import trainer # Placeholder
    class DummyTrainer: # Placeholder for your trainer class
        def __init__(self, model, train_loader, test_loader, criterion, optimizer, device):
            self.model = model
            self.train_loader = train_loader
            self.test_loader = test_loader
            self.criterion = criterion
            self.optimizer = optimizer
            self.device = device
            print("Trainer initialized.")

        def train(self, num_epochs):
            print(f"\n--- Starting Dummy Training for {num_epochs} epochs ---")
            self.model.train() # Set model to training mode
            for epoch in range(num_epochs):
                print(f"Epoch {epoch+1}/{num_epochs}")
                for i, (inputs, labels) in enumerate(self.train_loader):
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs) # Forward pass

                    # Handle potential tuple output (if aux_logits=True)
                    if isinstance(outputs, GoogLeNetOutputs):
                        # Combine losses (example: weighted sum)
                        loss_main = self.criterion(outputs.logits, labels)
                        loss_aux1 = self.criterion(outputs.aux_logits1, labels)
                        loss_aux2 = self.criterion(outputs.aux_logits2, labels)
                        loss = loss_main + 0.3 * loss_aux1 + 0.3 * loss_aux2 # Common weighting
                    else: # Only main logits returned
                        loss = self.criterion(outputs, labels)

                    loss.backward()
                    self.optimizer.step()

                    if (i + 1) % 2 == 0: # Print less frequently
                         print(f"  Batch {i+1}/{len(self.train_loader)}, Loss: {loss.item():.4f}")
                # Add evaluation step here if needed
            print("--- Dummy Training Complete ---")

    # Instantiate the trainer
    myTrainer = DummyTrainer(  # Replace with your actual trainer.Trainer
        model=myModel,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=myModel.criterion, # Use criterion from the model
        optimizer=myModel.optimizer, # Use optimizer from the model
        device=device
    )

    print(f"\nUsing device: {device}")
    myModel.print_num_params() # Use the built-in method

    # Start training
    myTrainer.train(num_epochs=CONFIGURE.num_epochs)

    # --- Example: Creating without pretrained weights ---
    print("\n--- Creating GoogLeNetCustom (From Scratch) ---")
    model_scratch = GoogLeNetCustom(
        num_classes=CONFIGURE.num_classes,
        lr=CONFIGURE.learning_rate,
        pretrained=False,
        init_weights=True # Explicitly initialize
    ).to(device=device)
    model_scratch.print_num_params()
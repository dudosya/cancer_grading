# CBAM_2_ChannelActivation.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Activation Function Mapping ---
# Define available activations here for clarity and easy extension
_ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,  # Also known as Swish
    "mish": nn.Mish,
    "hardswish": nn.Hardswish,
}

class ChannelAttention_2_ChannelActivation(nn.Module): # Name updated
    """
    Channel Attention Module (CAM) part of CBAM.
    **Modification:** Allows specifying the activation function in the shared MLP.
    """
    def __init__(self, gate_channels, reduction_ratio=16, activation_type="relu"): # Added activation_type
        """
        Initializes the ChannelAttention module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the MLP channels. Default is 16.
            activation_type (str): Type of activation function to use in the MLP.
                                   Options: "relu", "gelu", "silu", "mish", "hardswish".
                                   Default is "relu".
        """
        super(ChannelAttention_2_ChannelActivation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(1, gate_channels // reduction_ratio)

        # --- Select Activation Function ---
        activation_key = activation_type.lower()
        if activation_key not in _ACTIVATION_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}. Available: {list(_ACTIVATION_MAP.keys())}")
        ActivationLayer = _ACTIVATION_MAP[activation_key]
        # Handle inplace argument specifically for ReLU if desired
        if activation_key == "relu":
            activation_instance = ActivationLayer(inplace=True)
        else:
            activation_instance = ActivationLayer()
        print(f"  Initializing ChannelAttention_2_ChannelActivation with activation: {activation_key}") # Added print

        # --- Define Shared MLP with selected activation ---
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            activation_instance, # Use the selected activation layer
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_2_ChannelActivation(nn.Module): # Name updated for consistency
    """
    Spatial Attention Module (SAM) part of CBAM.
    (Identical to original SpatialAttention, renamed for consistency within this file)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention_2_ChannelActivation, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing SpatialAttention_2_ChannelActivation (no change)") # Optional print

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_2_ChannelActivation(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Allows specifying the activation function in the Channel Attention MLP.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7,
                 channel_activation_type="relu"): # Added channel_activation_type
        """
        Initializes the CBAM module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
            spatial_kernel_size (int): Kernel size for the Spatial Attention convolution.
            channel_activation_type (str): Activation function for the Channel Attention MLP.
                                           Default is "relu". **This is the modified parameter.**
        """
        super(CBAM_2_ChannelActivation, self).__init__()
        print(f"Initializing CBAM_2_ChannelActivation with gate_channels={gate_channels}, channel_activation={channel_activation_type}") # Added print
        # Use the renamed internal modules
        # Pass the activation type down to ChannelAttention
        self.channel_attention = ChannelAttention_2_ChannelActivation(
            gate_channels,
            reduction_ratio,
            activation_type=channel_activation_type # Pass the argument
        )
        self.spatial_attention = SpatialAttention_2_ChannelActivation(spatial_kernel_size)

    def forward(self, x):
        """ Forward pass - No change needed here """
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Broadcasting
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Broadcasting
        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_2_ChannelActivation ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    # --- Test with default activation (ReLU) ---
    print("\nTesting with default channel activation (ReLU):")
    cbam_relu = CBAM_2_ChannelActivation(gate_channels=64, channel_activation_type="relu")
    # print("CBAM Module (ReLU):\n", cbam_relu) # Optional print
    output_relu = cbam_relu(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (ReLU): {output_relu.shape}")
    assert dummy_input.shape == output_relu.shape, "Output shape (ReLU) mismatch!"
    print("Successfully tested CBAM (ReLU)")

    # --- Test with GELU ---
    print("\nTesting with channel activation GELU:")
    cbam_gelu = CBAM_2_ChannelActivation(gate_channels=64, channel_activation_type="gelu")
    # print("CBAM Module (GELU):\n", cbam_gelu) # Optional print
    output_gelu = cbam_gelu(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (GELU): {output_gelu.shape}")
    assert dummy_input.shape == output_gelu.shape, "Output shape (GELU) mismatch!"
    print("Successfully tested CBAM (GELU)")

    # --- Test with SiLU ---
    print("\nTesting with channel activation SiLU:")
    cbam_silu = CBAM_2_ChannelActivation(gate_channels=64, channel_activation_type="silu")
    # print("CBAM Module (SiLU):\n", cbam_silu) # Optional print
    output_silu = cbam_silu(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (SiLU): {output_silu.shape}")
    assert dummy_input.shape == output_silu.shape, "Output shape (SiLU) mismatch!"
    print("Successfully tested CBAM (SiLU)")

    # --- Test with Mish ---
    print("\nTesting with channel activation Mish:")
    cbam_mish = CBAM_2_ChannelActivation(gate_channels=64, channel_activation_type="mish")
    # print("CBAM Module (Mish):\n", cbam_mish) # Optional print
    output_mish = cbam_mish(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (Mish): {output_mish.shape}")
    assert dummy_input.shape == output_mish.shape, "Output shape (Mish) mismatch!"
    print("Successfully tested CBAM (Mish)")

    # --- Test with Invalid Activation ---
    print("\nTesting with invalid activation type:")
    try:
        cbam_invalid = CBAM_2_ChannelActivation(gate_channels=64, channel_activation_type="unknown")
    except ValueError as e:
        print(f"Caught expected error: {e}")
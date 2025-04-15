# CBAM_4_NoChannelBottleneck.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_4_NoChannelBottleneck(nn.Module): # Name updated
    """
    Channel Attention Module (CAM) part of CBAM.
    **Modification:** Removes the bottleneck (dimensionality reduction) in the shared MLP.
                     The MLP is now just a single 1x1 Conv layer.
                     The `reduction_ratio` parameter is ignored by this module.
    """
    def __init__(self, gate_channels, reduction_ratio=16, **kwargs): # reduction_ratio kept for API compatibility but unused
        """
        Initializes the ChannelAttention module without bottleneck.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Kept for API compatibility but **not used**.
            **kwargs: Absorbs any other potential arguments passed.
        """
        super(ChannelAttention_4_NoChannelBottleneck, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # --- Modified Shared MLP (No Bottleneck) ---
        # Simply map channels to themselves using a 1x1 Conv
        self.shared_mlp = nn.Conv2d(gate_channels, gate_channels, kernel_size=1, bias=False)
        # Note: No ReLU is typically used in this direct mapping variant,
        # mimicking ECA-Net's approach more closely. If desired, it could be added.

        self.sigmoid = nn.Sigmoid()
        print(f"  Initializing ChannelAttention_4_NoChannelBottleneck (No Bottleneck/Reduction)") # Added print

    def forward(self, x):
        """
        Forward pass for Channel Attention.
        """
        # Apply pooling
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Add the outputs from the simplified MLP and apply sigmoid
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_4_NoChannelBottleneck(nn.Module): # Name updated for consistency
    """
    Spatial Attention Module (SAM) part of CBAM.
    (Identical to original SpatialAttention, renamed for consistency within this file)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention_4_NoChannelBottleneck, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing SpatialAttention_4_NoChannelBottleneck (no change)") # Optional print

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_4_NoChannelBottleneck(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Uses Channel Attention without the MLP bottleneck.
                     The `reduction_ratio` parameter is ignored.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7): # reduction_ratio kept for API consistency
        """
        Initializes the CBAM module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Kept for API compatibility but **not used** by the
                                   ChannelAttention sub-module in this version.
            spatial_kernel_size (int): Kernel size for the Spatial Attention convolution.
        """
        super(CBAM_4_NoChannelBottleneck, self).__init__()
        print(f"Initializing CBAM_4_NoChannelBottleneck with gate_channels={gate_channels} (Reduction Ratio Ignored)") # Added print
        # Use the renamed internal modules
        # Instantiate ChannelAttention variant without bottleneck
        self.channel_attention = ChannelAttention_4_NoChannelBottleneck(gate_channels) # No need to pass reduction_ratio
        self.spatial_attention = SpatialAttention_4_NoChannelBottleneck(spatial_kernel_size)

    def forward(self, x):
        """ Forward pass - No change needed here """
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Broadcasting
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Broadcasting
        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_4_NoChannelBottleneck ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    print("\nTesting CBAM without channel bottleneck:")
    # reduction_ratio is passed but ignored internally by ChannelAttention_4
    cbam_no_bottleneck = CBAM_4_NoChannelBottleneck(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)

    output = cbam_no_bottleneck(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (No Bottleneck): {output.shape}")
    assert dummy_input.shape == output.shape, "Output shape (No Bottleneck) mismatch!"
    print("Successfully tested CBAM (No Bottleneck)")

    # --- Verify parameter count difference (Optional) ---
    print("\nComparing parameter counts:")
    # Import original for comparison (assuming it's accessible)
    try:
        from .CBAM import CBAM as CBAM_Original
        cbam_orig = CBAM_Original(gate_channels=64, reduction_ratio=16)
        params_orig = sum(p.numel() for p in cbam_orig.parameters() if p.requires_grad)
        print(f"Original CBAM Params: {params_orig:,}")
    except ImportError:
        print("Could not import original CBAM for comparison.")

    params_no_bottleneck = sum(p.numel() for p in cbam_no_bottleneck.parameters() if p.requires_grad)
    print(f"CBAM No Bottleneck Params: {params_no_bottleneck:,}")
    # Expect params_no_bottleneck to be lower due to simpler channel MLP
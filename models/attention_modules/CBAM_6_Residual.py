# CBAM_6_Residual.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_6_Residual(nn.Module): # Name updated for consistency
    """
    Channel Attention Module (CAM) part of CBAM.
    (Identical to original ChannelAttention, renamed for consistency within this file)
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelAttention_6_Residual, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, gate_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing ChannelAttention_6_Residual (no change)") # Optional

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_6_Residual(nn.Module): # Name updated for consistency
    """
    Spatial Attention Module (SAM) part of CBAM.
    (Identical to original SpatialAttention, renamed for consistency within this file)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention_6_Residual, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing SpatialAttention_6_Residual (no change)") # Optional

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_6_Residual(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Adds a residual connection, returning `x + attention_refined_x`.

    Applies Channel Attention followed by Spatial Attention sequentially,
    then adds the result back to the original input.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        Initializes the Residual CBAM module. Arguments are the same as original CBAM.
        """
        super(CBAM_6_Residual, self).__init__()
        print(f"Initializing CBAM_6_Residual with gate_channels={gate_channels} (Adds Residual Connection)") # Added print
        # Use the renamed internal modules
        self.channel_attention = ChannelAttention_6_Residual(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention_6_Residual(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass for Residual CBAM.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Feature map after applying attention and adding the
                          original input back (`x + refined_x`),
                          same shape as input (B, C, H, W).
        """
        # Store original input for residual connection
        identity = x

        # Apply Channel Attention sequentially
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Broadcasting

        # Apply Spatial Attention sequentially
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Broadcasting

        # --- Add Residual Connection ---
        output = identity + x_refined

        return output

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_6_Residual ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    print("\nTesting CBAM with residual connection:")
    cbam_residual = CBAM_6_Residual(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)

    # Verify internal modules exist
    print("Internal Channel Attention:", hasattr(cbam_residual, 'channel_attention'))
    print("Internal Spatial Attention:", hasattr(cbam_residual, 'spatial_attention'))

    output = cbam_residual(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape (Residual): {output.shape}")
    assert dummy_input.shape == output.shape, "Output shape (Residual) mismatch!"
    print("Successfully tested CBAM (Residual)")

    # --- Optional: Compare parameter counts (should be identical to original) ---
    print("\nComparing parameter counts:")
    try:
        from .CBAM import CBAM as CBAM_Original # Assuming original is accessible
        cbam_orig = CBAM_Original(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
        params_orig = sum(p.numel() for p in cbam_orig.parameters() if p.requires_grad)
        print(f"Original CBAM Params: {params_orig:,}")
    except ImportError:
        print("Could not import original CBAM for comparison.")

    params_residual = sum(p.numel() for p in cbam_residual.parameters() if p.requires_grad)
    print(f"CBAM Residual Params: {params_residual:,}")
    # Expect params_residual to be the same as original CBAM
    if 'params_orig' in locals():
        assert params_orig == params_residual, "Parameter counts should be identical!"
        print("Parameter counts match original CBAM, as expected.")
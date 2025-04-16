# CBAM_5_Parallel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_5_Parallel(nn.Module): # Name updated for consistency
    """
    Channel Attention Module (CAM) part of CBAM.
    (Identical to original ChannelAttention, renamed for consistency within this file)
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelAttention_5_Parallel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, gate_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing ChannelAttention_5_Parallel (no change)") # Optional

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_5_Parallel(nn.Module): # Name updated for consistency
    """
    Spatial Attention Module (SAM) part of CBAM.
    (Identical to original SpatialAttention, renamed for consistency within this file)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention_5_Parallel, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing SpatialAttention_5_Parallel (no change)") # Optional

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_5_Parallel(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Computes Channel and Spatial Attention maps in parallel
                     from the original input `x`, combines them, and then applies
                     the combined map to `x`.

    Original CBAM applies Channel Attention, then Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        Initializes the Parallel CBAM module. Arguments are the same as original CBAM.
        """
        super(CBAM_5_Parallel, self).__init__()
        print(f"Initializing CBAM_5_Parallel with gate_channels={gate_channels} (Parallel Application)") # Added print
        # Use the renamed internal modules
        self.channel_attention = ChannelAttention_5_Parallel(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention_5_Parallel(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass for Parallel CBAM.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Refined feature map after applying combined parallel attention,
                          same shape as input (B, C, H, W).
        """
        # 1. Compute Channel Attention Map from original x
        # Output shape: (B, C, 1, 1)
        channel_map = self.channel_attention(x)

        # 2. Compute Spatial Attention Map from original x
        # Output shape: (B, 1, H, W)
        spatial_map = self.spatial_attention(x)

        # 3. Combine the maps (element-wise multiplication using broadcasting)
        # (B, C, 1, 1) * (B, 1, H, W) -> (B, C, H, W)
        combined_map = channel_map * spatial_map

        # 4. Apply the combined attention map to the original input x
        # (B, C, H, W) * (B, C, H, W) -> (B, C, H, W)
        x_refined = x * combined_map

        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_5_Parallel ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    print("\nTesting CBAM with parallel attention computation:")
    cbam_parallel = CBAM_5_Parallel(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)

    # Verify internal modules exist
    print("Internal Channel Attention:", hasattr(cbam_parallel, 'channel_attention'))
    print("Internal Spatial Attention:", hasattr(cbam_parallel, 'spatial_attention'))

    output = cbam_parallel(dummy_input)
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape (Parallel): {output.shape}")
    assert dummy_input.shape == output.shape, "Output shape (Parallel) mismatch!"
    print("Successfully tested CBAM (Parallel)")

    # --- Optional: Compare parameter counts (should be identical to original) ---
    print("\nComparing parameter counts:")
    try:
        from .CBAM import CBAM as CBAM_Original # Assuming original is accessible
        cbam_orig = CBAM_Original(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
        params_orig = sum(p.numel() for p in cbam_orig.parameters() if p.requires_grad)
        print(f"Original CBAM Params: {params_orig:,}")
    except ImportError:
        print("Could not import original CBAM for comparison.")

    params_parallel = sum(p.numel() for p in cbam_parallel.parameters() if p.requires_grad)
    print(f"CBAM Parallel Params: {params_parallel:,}")
    # Expect params_parallel to be the same as original CBAM
    if 'params_orig' in locals():
       assert params_orig == params_parallel, "Parameter counts should be identical!"
       print("Parameter counts match original CBAM, as expected.")
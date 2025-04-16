# CBAM_7_SpatialDWConv.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_7_SpatialDWConv(nn.Module): # Name updated for consistency
    """
    Channel Attention Module (CAM) part of CBAM.
    (Identical to original ChannelAttention, renamed for consistency within this file)
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelAttention_7_SpatialDWConv, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, gate_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing ChannelAttention_7_SpatialDWConv (no change)") # Optional

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_7_SpatialDWConv(nn.Module): # Name updated
    """
    Spatial Attention Module (SAM) part of CBAM.
    **Modification:** Replaces the standard convolution with a
                     Depthwise-Separable Convolution for efficiency.
    """
    def __init__(self, kernel_size=7):
        """
        Initializes the SpatialAttention module with Depthwise-Separable Conv.

        Args:
            kernel_size (int): Kernel size for the depthwise convolution layer. Default is 7.
        """
        super(SpatialAttention_7_SpatialDWConv, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        padding = (kernel_size - 1) // 2
        in_channels = 2 # From concatenated avg and max pooling

        # --- Define Depthwise-Separable Convolution ---
        self.depthwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels, # Depthwise must have same in/out channels
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels, # groups=in_channels makes it depthwise
            bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1, # Pointwise maps to the final single channel
            kernel_size=1,
            padding=0,
            bias=False
        )
        # Combine into a sequential block
        self.conv = nn.Sequential(
            self.depthwise_conv,
            self.pointwise_conv
        )
        # --- End Definition ---

        self.sigmoid = nn.Sigmoid()
        print(f"  Initializing SpatialAttention_7_SpatialDWConv with kernel_size={kernel_size} (using Depthwise-Separable Conv)") # Added print


    def forward(self, x):
        """
        Forward pass for Spatial Attention. Uses the DW-Separable conv block.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1) # Shape: (B, 2, H, W)

        # Apply Depthwise-Separable convolution block
        spatial_att = self.sigmoid(self.conv(pooled_maps)) # Shape: (B, 1, H, W)
        return spatial_att


class CBAM_7_SpatialDWConv(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Uses Depthwise-Separable Convolution in the Spatial Attention module.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        Initializes the CBAM module with DW-Separable Spatial Conv.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
            spatial_kernel_size (int): Kernel size for the Spatial Attention depthwise conv.
        """
        super(CBAM_7_SpatialDWConv, self).__init__()
        print(f"Initializing CBAM_7_SpatialDWConv with gate_channels={gate_channels} (Spatial DW-Separable Conv)") # Added print
        # Use the renamed internal modules
        self.channel_attention = ChannelAttention_7_SpatialDWConv(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention_7_SpatialDWConv(spatial_kernel_size) # This now uses DW-Conv

    def forward(self, x):
        """ Forward pass - No change needed here """
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Broadcasting
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Broadcasting
        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_7_SpatialDWConv ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    print("\nTesting CBAM with Depthwise-Separable Spatial Convolution (k=7):")
    cbam_dw_k7 = CBAM_7_SpatialDWConv(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
    output_dw_k7 = cbam_dw_k7(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (DWConv k=7): {output_dw_k7.shape}")
    assert dummy_input.shape == output_dw_k7.shape, "Output shape (DWConv k=7) mismatch!"
    print("Successfully tested CBAM (DWConv k=7)")

    print("\nTesting CBAM with Depthwise-Separable Spatial Convolution (k=3):")
    cbam_dw_k3 = CBAM_7_SpatialDWConv(gate_channels=64, reduction_ratio=16, spatial_kernel_size=3)
    output_dw_k3 = cbam_dw_k3(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape (DWConv k=3): {output_dw_k3.shape}")
    assert dummy_input.shape == output_dw_k3.shape, "Output shape (DWConv k=3) mismatch!"
    print("Successfully tested CBAM (DWConv k=3)")

    # --- Compare parameter counts ---
    print("\nComparing parameter counts (k=7):")
    try:
        from .CBAM import CBAM as CBAM_Original # Assuming original is accessible
        cbam_orig_k7 = CBAM_Original(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
        params_orig_k7 = sum(p.numel() for p in cbam_orig_k7.parameters() if p.requires_grad)
        print(f"Original CBAM (k=7) Params: {params_orig_k7:,}") # Expect 610
    except ImportError:
        print("Could not import original CBAM for comparison.")

    params_dw_k7 = sum(p.numel() for p in cbam_dw_k7.parameters() if p.requires_grad)
    print(f"CBAM DW-Conv (k=7) Params: {params_dw_k7:,}") # Expect 512(Channel) + (2*1*7*7 + 2*1*1*1) = 512 + 98 + 2 = 612 ??? Let's recalculate
    # DW-Conv k=7:
    #   Depthwise(2,2,k=7,g=2): 2 * 1 * 7 * 7 = 98
    #   Pointwise(2,1,k=1):    2 * 1 * 1 * 1 = 2
    #   Total Spatial: 98 + 2 = 100
    # Total CBAM: 512 (Channel) + 100 (Spatial) = 612
    # So, slightly MORE params for k=7 DWConv because the Pointwise adds params.

    print("\nComparing parameter counts (k=3):")
    try:
        # Need to instantiate original with k=3 for fair comparison
        cbam_orig_k3 = CBAM_Original(gate_channels=64, reduction_ratio=16, spatial_kernel_size=3)
        params_orig_k3 = sum(p.numel() for p in cbam_orig_k3.parameters() if p.requires_grad)
        # Standard Conv k=3: 2 * 1 * 3 * 3 = 18. Total = 512 + 18 = 530
        print(f"Original CBAM (k=3) Params: {params_orig_k3:,}")
    except ImportError:
        print("Could not import original CBAM for comparison.")

    params_dw_k3 = sum(p.numel() for p in cbam_dw_k3.parameters() if p.requires_grad)
    # DW-Conv k=3:
    #   Depthwise(2,2,k=3,g=2): 2 * 1 * 3 * 3 = 18
    #   Pointwise(2,1,k=1):    2 * 1 * 1 * 1 = 2
    #   Total Spatial: 18 + 2 = 20
    # Total CBAM: 512 (Channel) + 20 (Spatial) = 532
    print(f"CBAM DW-Conv (k=3) Params: {params_dw_k3:,}")
    # So, DW-Separable actually has slightly MORE parameters in this specific low-channel spatial case.
    # The main benefit is usually FLOPs reduction.

    if 'params_orig_k7' in locals() and 'params_dw_k7' in locals():
        print(f"Param difference (k=7): {params_dw_k7 - params_orig_k7}")
    if 'params_orig_k3' in locals() and 'params_dw_k3' in locals():
        print(f"Param difference (k=3): {params_dw_k3 - params_orig_k3}")
# CBAM_1_SpatialKernel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_1_SpatialKernel(nn.Module): # Name updated for consistency
    """
    Channel Attention Module (CAM) part of CBAM.
    (Identical to original ChannelAttention, renamed for consistency within this file)
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelAttention_1_SpatialKernel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduced_channels = max(1, gate_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention_1_SpatialKernel(nn.Module): # Name updated
    """
    Spatial Attention Module (SAM) part of CBAM.
    **Modification:** Allows specifying the kernel size for the convolution.
    """
    def __init__(self, kernel_size=7): # kernel_size is now the configurable parameter
        """
        Initializes the SpatialAttention module.

        Args:
            kernel_size (int): Kernel size for the convolution layer. Default is 7.
        """
        super(SpatialAttention_1_SpatialKernel, self).__init__()

        # Ensure padding maintains spatial dimensions: padding = (kernel_size - 1) // 2
        assert kernel_size % 2 == 1, "Kernel size must be odd for 'same' padding"
        padding = (kernel_size - 1) // 2

        # Input channels = 2 (one from avg pool, one from max pool)
        # Output channels = 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        print(f"  Initializing SpatialAttention_1_SpatialKernel with kernel_size={kernel_size}, padding={padding}") # Added print

    def forward(self, x):
        """
        Forward pass for Spatial Attention.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Spatial attention map (B, 1, H, W), scaled by sigmoid.
        """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_1_SpatialKernel(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Allows specifying the spatial kernel size.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7): # spatial_kernel_size exposed
        """
        Initializes the CBAM module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
                                   Default is 16.
            spatial_kernel_size (int): Kernel size for the Spatial Attention convolution.
                                       Default is 7. **This is the modified parameter.**
        """
        super(CBAM_1_SpatialKernel, self).__init__()
        print(f"Initializing CBAM_1_SpatialKernel with gate_channels={gate_channels}, spatial_kernel_size={spatial_kernel_size}") # Added print
        # Use the renamed internal modules
        self.channel_attention = ChannelAttention_1_SpatialKernel(gate_channels, reduction_ratio)
        # Pass the specific kernel size to the spatial attention module
        self.spatial_attention = SpatialAttention_1_SpatialKernel(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass for CBAM.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Refined feature map after applying CAM and SAM,
                          same shape as input (B, C, H, W).
        """
        # Apply Channel Attention
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Use broadcasting

        # Apply Spatial Attention
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Use broadcasting

        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_1_SpatialKernel ---")

    # --- Test with default kernel size (7x7) ---
    print("\nTesting with default spatial kernel size (7):")
    dummy_input_7 = torch.randn(4, 64, 32, 32)
    cbam_block_7 = CBAM_1_SpatialKernel(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
    # print("CBAM Module (k=7):\n", cbam_block_7) # Optional: print module structure
    output_7 = cbam_block_7(dummy_input_7)
    print(f"Input shape:  {dummy_input_7.shape}")
    print(f"Output shape (k=7): {output_7.shape}")
    assert dummy_input_7.shape == output_7.shape, "Output shape (k=7) doesn't match input shape!"
    print("Successfully tested CBAM (k=7): Output shape matches input shape.")

    # --- Test with a different kernel size (e.g., 3x3) ---
    print("\nTesting with spatial kernel size 3:")
    dummy_input_3 = torch.randn(4, 64, 32, 32)
    cbam_block_3 = CBAM_1_SpatialKernel(gate_channels=64, reduction_ratio=16, spatial_kernel_size=3)
    # print("CBAM Module (k=3):\n", cbam_block_3) # Optional: print module structure
    output_3 = cbam_block_3(dummy_input_3)
    print(f"Input shape:  {dummy_input_3.shape}")
    print(f"Output shape (k=3): {output_3.shape}")
    assert dummy_input_3.shape == output_3.shape, "Output shape (k=3) doesn't match input shape!"
    print("Successfully tested CBAM (k=3): Output shape matches input shape.")

    # --- Test with another kernel size (e.g., 5x5) ---
    print("\nTesting with spatial kernel size 5:")
    dummy_input_5 = torch.randn(4, 64, 32, 32)
    cbam_block_5 = CBAM_1_SpatialKernel(gate_channels=64, reduction_ratio=16, spatial_kernel_size=5)
    # print("CBAM Module (k=5):\n", cbam_block_5) # Optional: print module structure
    output_5 = cbam_block_5(dummy_input_5)
    print(f"Input shape:  {dummy_input_5.shape}")
    print(f"Output shape (k=5): {output_5.shape}")
    assert dummy_input_5.shape == output_5.shape, "Output shape (k=5) doesn't match input shape!"
    print("Successfully tested CBAM (k=5): Output shape matches input shape.")
import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    """
    Channel Attention Module (CAM) part of CBAM.

    Applies Average and Max Pooling across spatial dimensions, feeds through
    a shared MLP, combines, and applies sigmoid to get channel attention weights.
    """
    def __init__(self, gate_channels, reduction_ratio=16):
        """
        Initializes the ChannelAttention module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the MLP channels. Default is 16.
        """
        super(ChannelAttention, self).__init__()
        # Use Adaptive Pooling to handle varying spatial dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP: Conv -> ReLU -> Conv
        # Using Conv2d with kernel_size 1 acts like a Linear layer on channels
        reduced_channels = max(1, gate_channels // reduction_ratio) # Ensure reduced_channels is at least 1
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for Channel Attention.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Channel attention map (B, C, 1, 1), scaled by sigmoid.
        """
        # Apply pooling
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # Add the outputs and apply sigmoid
        channel_att = self.sigmoid(avg_out + max_out)
        return channel_att


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module (SAM) part of CBAM.

    Applies Average and Max Pooling across the channel dimension, concatenates
    them, applies a convolution, and uses sigmoid to get spatial attention weights.
    """
    def __init__(self, kernel_size=7):
        """
        Initializes the SpatialAttention module.

        Args:
            kernel_size (int): Kernel size for the convolution layer. Default is 7.
        """
        super(SpatialAttention, self).__init__()

        # Ensure padding maintains spatial dimensions: padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        # Input channels = 2 (one from avg pool, one from max pool)
        # Output channels = 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass for Spatial Attention.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W). This should be the
                              output after applying Channel Attention.

        Returns:
            torch.Tensor: Spatial attention map (B, 1, H, W), scaled by sigmoid.
        """
        # Apply pooling across the channel dimension
        # Keepdim=True to maintain dimensions (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # Max returns values and indices

        # Concatenate along the channel dimension
        pooled_maps = torch.cat([avg_out, max_out], dim=1) # Shape: (B, 2, H, W)

        # Apply convolution and sigmoid
        spatial_att = self.sigmoid(self.conv(pooled_maps)) # Shape: (B, 1, H, W)
        return spatial_att


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Applies Channel Attention followed by Spatial Attention sequentially to
    refine input feature maps.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7):
        """
        Initializes the CBAM module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
                                   Default is 16.
            spatial_kernel_size (int): Kernel size for the Spatial Attention convolution.
                                       Default is 7.
        """
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(gate_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        """
        Forward pass for CBAM.

        Args:
            x (torch.Tensor): Input feature map (B, C, H, W).

        Returns:
            torch.Tensor: Refined feature map after applying CAM and SAM,
                          same shape as input (B, C, H, W).
        """
        # Apply Channel Attention: x_refined = x * channel_attention(x)
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map.expand_as(x) # Broadcast (B,C,1,1) to (B,C,H,W)

        # Apply Spatial Attention: final_output = x_refined * spatial_attention(x_refined)
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map.expand_as(x_after_channel_att) # Broadcast (B,1,H,W) to (B,C,H,W)

        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    # Create a dummy input tensor
    # Batch size=4, Channels=64, Height=32, Width=32
    dummy_input = torch.randn(4, 64, 32, 32)

    # Instantiate CBAM module
    cbam_block = CBAM(gate_channels=64, reduction_ratio=16, spatial_kernel_size=7)
    print("CBAM Module:\n", cbam_block)

    # Pass input through CBAM
    output = cbam_block(dummy_input)

    # Print input and output shapes to verify
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Check if output shape is the same as input shape
    assert dummy_input.shape == output.shape, "Output shape doesn't match input shape!"
    print("\nSuccessfully tested CBAM: Output shape matches input shape.")

    # --- Test individual components ---
    print("\nTesting Channel Attention individually:")
    channel_att_module = ChannelAttention(gate_channels=64)
    channel_map = channel_att_module(dummy_input)
    print(f"Channel Attention output map shape: {channel_map.shape}") # Should be [4, 64, 1, 1]

    print("\nTesting Spatial Attention individually:")
    spatial_att_module = SpatialAttention(kernel_size=7)
    # SpatialAttention expects the channel-refined input, but we can test its shape logic
    spatial_map = spatial_att_module(dummy_input * channel_map) # Use dummy refined input
    print(f"Spatial Attention output map shape: {spatial_map.shape}") # Should be [4, 1, 32, 32]
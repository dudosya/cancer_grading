# CBAM_3_CombineAvgMaxAdd.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention_3_CombineAvgMaxAdd(nn.Module): # Name updated
    """
    Channel Attention Module (CAM) part of CBAM.
    **Modification:** Allows specifying how AvgPool and MaxPool results are combined ('add' or 'max').
    """
    def __init__(self, gate_channels, reduction_ratio=16, combination_method="add"): # Added combination_method
        """
        Initializes the ChannelAttention module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the MLP channels. Default is 16.
            combination_method (str): How to combine AvgPool and MaxPool features
                                      after the MLP. Options: "add", "max". Default is "add".
        """
        super(ChannelAttention_3_CombineAvgMaxAdd, self).__init__()

        self.combination_method = combination_method.lower()
        if self.combination_method not in ["add", "max"]:
            raise ValueError(f"Unsupported combination method: {combination_method}. Choose 'add' or 'max'.")

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        reduced_channels = max(1, gate_channels // reduction_ratio)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(gate_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, gate_channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        print(f"  Initializing ChannelAttention_3_CombineAvgMaxAdd with combination: '{self.combination_method}'") # Added print

    def forward(self, x):
        """
        Forward pass for Channel Attention. Applies the selected combination method.
        """
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))

        # --- Apply selected combination method ---
        if self.combination_method == "add":
            combined = avg_out + max_out
        elif self.combination_method == "max":
            combined = torch.max(avg_out, max_out)
        else:
            # This case should not be reached due to check in __init__, but included for safety
            raise ValueError(f"Internal error: Invalid combination method '{self.combination_method}' encountered.")

        channel_att = self.sigmoid(combined)
        return channel_att


class SpatialAttention_3_CombineAvgMaxAdd(nn.Module): # Name updated for consistency
    """
    Spatial Attention Module (SAM) part of CBAM.
    (Identical to original SpatialAttention, renamed for consistency within this file)
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention_3_CombineAvgMaxAdd, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        # print(f"  Initializing SpatialAttention_3_CombineAvgMaxAdd (no change)") # Optional print

    def forward(self, x):
        """ Forward pass - No change needed here """
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled_maps = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.conv(pooled_maps))
        return spatial_att


class CBAM_3_CombineAvgMaxAdd(nn.Module): # Name updated
    """
    Convolutional Block Attention Module (CBAM).
    **Modification:** Allows specifying the combination method ('add' or 'max')
    in the Channel Attention module.

    Applies Channel Attention followed by Spatial Attention sequentially.
    """
    def __init__(self, gate_channels, reduction_ratio=16, spatial_kernel_size=7,
                 channel_combination_method="add"): # Added channel_combination_method
        """
        Initializes the CBAM module.

        Args:
            gate_channels (int): Number of channels in the input feature map.
            reduction_ratio (int): Reduction ratio for the Channel Attention MLP.
            spatial_kernel_size (int): Kernel size for the Spatial Attention convolution.
            channel_combination_method (str): How to combine features in Channel Attention.
                                             Options: "add", "max". Default is "add".
                                             **This is the modified parameter.**
        """
        super(CBAM_3_CombineAvgMaxAdd, self).__init__()
        print(f"Initializing CBAM_3_CombineAvgMaxAdd with gate_channels={gate_channels}, channel_combination='{channel_combination_method}'") # Added print
        # Use the renamed internal modules
        # Pass the combination method down to ChannelAttention
        self.channel_attention = ChannelAttention_3_CombineAvgMaxAdd(
            gate_channels,
            reduction_ratio,
            combination_method=channel_combination_method # Pass the argument
        )
        self.spatial_attention = SpatialAttention_3_CombineAvgMaxAdd(spatial_kernel_size)

    def forward(self, x):
        """ Forward pass - No change needed here """
        channel_att_map = self.channel_attention(x)
        x_after_channel_att = x * channel_att_map # Broadcasting
        spatial_att_map = self.spatial_attention(x_after_channel_att)
        x_refined = x_after_channel_att * spatial_att_map # Broadcasting
        return x_refined

# --- Example Usage (for testing within this file) ---
if __name__ == '__main__':
    print("--- Testing CBAM_3_CombineAvgMaxAdd ---")
    dummy_input = torch.randn(4, 64, 32, 32)

    # --- Test with default combination (add) ---
    print("\nTesting with default channel combination ('add'):")
    cbam_add = CBAM_3_CombineAvgMaxAdd(gate_channels=64, channel_combination_method="add")
    output_add = cbam_add(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape ('add'): {output_add.shape}")
    assert dummy_input.shape == output_add.shape, "Output shape ('add') mismatch!"
    print("Successfully tested CBAM ('add')")

    # --- Test with max combination ---
    print("\nTesting with channel combination 'max':")
    cbam_max = CBAM_3_CombineAvgMaxAdd(gate_channels=64, channel_combination_method="max")
    output_max = cbam_max(dummy_input)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape ('max'): {output_max.shape}")
    assert dummy_input.shape == output_max.shape, "Output shape ('max') mismatch!"
    print("Successfully tested CBAM ('max')")

    # --- Test with Invalid Combination Method ---
    print("\nTesting with invalid combination method:")
    try:
        cbam_invalid = CBAM_3_CombineAvgMaxAdd(gate_channels=64, channel_combination_method="multiply")
    except ValueError as e:
        print(f"Caught expected error: {e}")
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CvtForImageClassification, CvtConfig, AutoFeatureExtractor


class CvTModelHF(nn.Module):  # HF for Hugging Face
    def __init__(self, num_classes=4, lr=0.001, model_name="microsoft/cvt-13"):
        super(CvTModelHF, self).__init__()

        # Load the pre-trained CvT model from Hugging Face Transformers
        self.feature_extractor = CvtForImageClassification.from_pretrained(
            model_name,  # Use a specific pre-trained model
            num_labels=num_classes,  # Directly set the number of labels here!
            ignore_mismatched_sizes=True  # Important: Allow changing the classifier
        )

        #  The transformers library *already* handles replacing the classifier
        #  when we set `num_labels`.  We don't need to do it manually.

        # Define the criterion and optimizer (you can customize these)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=lr, momentum=0.9)

    def forward(self, x, labels=None):
        # Forward pass through the model.  The transformers models return
        # a ModelOutput object, and the logits are in the `logits` attribute.
        outputs = self.feature_extractor(x, labels=labels)
        logits = outputs.logits
        return logits, logits  # Consistent return format

    def print_num_params(self):
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params_millions = total_params / 1e6
        print(f"Total trainable params: {total_params_millions:.3g} M")




# Example usage
if __name__ == '__main__':
    # --- Using the model directly ---

    # Check for CUDA availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("CUDA device not found. Using CPU.")


    model = CvTModelHF(num_classes=4, lr=0.001).to(device)
    model.print_num_params()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)  # Typical CvT input size
    dummy_input = dummy_input.to(device)  # Move the input to the same device as the model
    output, _ = model(dummy_input)
    print("Output shape:", output.shape)


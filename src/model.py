import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Config

class ResNet50Backbone(nn.Module):
    """
    Visual Backbone: Modified ResNet-50 for grayscale X-ray images.
    """
    def __init__(self, pretrained=True):
        super(ResNet50Backbone, self).__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)

        # Modify first layer to accept grayscale images (1 channel)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.mean(resnet50.conv1.weight, dim=1, keepdim=True))

        # Use ResNet layers as feature extractor
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[1:-2])  # Remove last two layers
        self.flatten = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.feature_extractor(x)
        x = self.flatten(x)
        x = torch.flatten(x, 1)
        return x

class RadTexModel(nn.Module):
    """
    Full RadTex-inspired model for:
      1. Image-to-Text Report Generation (Transformer Decoder with Cross-Attention)
      2. Binary Classification (FC Head)
    """
    def __init__(self, vocab_size, num_classes=1, pretrained_backbone=True):
        super(RadTexModel, self).__init__()
        self.visual_backbone = ResNet50Backbone(pretrained=pretrained_backbone)

        # Linear Projection to match Transformer embedding size
        self.feature_projection = nn.Linear(2048, 768)  # GPT-2 hidden size

        # ✅ Transformer Decoder with Cross-Attention
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=768,
            n_layer=6,
            n_head=8,
            bos_token_id=0,
            eos_token_id=1,
            add_cross_attention=True  # ✅ FIX: Enable Cross-Attention for Encoder-Decoder Setup
        )
        self.text_decoder = GPT2LMHeadModel(config)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, images, text_inputs=None):
        """
        Args:
            images (Tensor): X-ray images (batch, 1, 256, 256)
            text_inputs (Tensor): Tokenized text input for Transformer (batch, seq_len)

        Returns:
            classification_output: Binary label prediction (batch, 1)
            generated_text: Generated free-text report (if text_inputs is provided)
        """
        # Extract visual features
        visual_features = self.visual_backbone(images)  # (batch, 2048)
        projected_features = self.feature_projection(visual_features)  # (batch, 768)

        # Text decoder: Generate report
        generated_text = None
        if text_inputs is not None:
            transformer_inputs = {
                "input_ids": text_inputs,
                "encoder_hidden_states": projected_features.unsqueeze(1)  # ✅ Now supported!
            }
            generated_text = self.text_decoder(**transformer_inputs).logits  # (batch, seq_len, vocab_size)

        # Binary classification output
        classification_output = self.classifier(visual_features)  # (batch, 1)

        return classification_output, generated_text

if __name__ == "__main__":
    # ✅ Test Model
    batch_size = 2
    dummy_images = torch.randn(batch_size, 1, 256, 256)  # Grayscale X-ray images
    dummy_text = torch.randint(0, 30522, (batch_size, 30))  # Simulated tokenized text (GPT-2 vocab size)

    model = RadTexModel(vocab_size=30522)  # GPT-2 vocab size
    classification_output, generated_text = model(dummy_images, dummy_text)

    print(f"Classification Output: {classification_output.shape}")  # Expected: (batch_size, 1)
    print(f"Generated Text Output: {generated_text.shape}")  # Expected: (batch_size, seq_len, vocab_size)

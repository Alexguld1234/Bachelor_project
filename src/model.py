import torch
import torch.nn as nn
import torchvision.models as models

class ChestXrayReportGenerator(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        """
        Args:
            vocab_size (int): Vocabulary size for text generation.
            embed_dim (int): Embedding dimension for text.
            hidden_dim (int): Hidden size of RNN decoder.
            num_layers (int): Number of GRU layers.
        """
        super(ChestXrayReportGenerator, self).__init__()

        # Image Encoder - Pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove classification layer
        self.feature_dim = resnet.fc.in_features  # Get last layer feature size

        # Fully Connected Layer for Classification (Pneumonia Prediction)
        self.classification_head = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),  # Binary output (Pneumonia or not)
            nn.Sigmoid()
        )

        # Text Decoder - GRU-based
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.gru = nn.GRU(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)  # Output layer for text generation

    def forward(self, image, captions):
        """
        Forward pass for multimodal model.
        Args:
            image (Tensor): Image input (batch_size, C, H, W).
            captions (Tensor): Tokenized captions (batch_size, seq_len).
        Returns:
            class_pred (Tensor): Pneumonia classification output.
            report_pred (Tensor): Generated text sequence output.
        """
        # Encode Image
        img_features = self.image_encoder(image)
        img_features = img_features.view(img_features.size(0), -1)  # Flatten

        # Pneumonia Classification
        class_pred = self.classification_head(img_features)

        # Decode Report (Text Generation)
        embedded_captions = self.embedding(captions)
        outputs, _ = self.gru(embedded_captions)
        report_pred = self.fc_out(outputs)  # Convert GRU outputs to vocab logits

        return class_pred, report_pred

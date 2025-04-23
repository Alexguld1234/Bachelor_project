import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Config

class ResNet50Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(torch.mean(resnet50.conv1.weight, dim=1, keepdim=True))
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[1:-2])
        self.flatten = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.feature_extractor(x)
        x = self.flatten(x)
        return torch.flatten(x, 1)

class RadTexModel(nn.Module):
    def __init__(self, vocab_size, num_classes=1, pretrained_backbone=True):
        super().__init__()
        self.visual_backbone = ResNet50Backbone(pretrained=pretrained_backbone)
        self.feature_projection = nn.Linear(2048, 768)

        config = GPT2Config.from_pretrained("gpt2")
        config.add_cross_attention = True
        config.vocab_size = vocab_size
        config.pad_token_id = config.eos_token_id
        config.bos_token_id = config.bos_token_id or config.eos_token_id

        self.text_decoder = GPT2LMHeadModel(config)
        self.text_decoder.resize_token_embeddings(vocab_size)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, images, text_inputs=None, generate=False, max_length=50):
        visual_features = self.visual_backbone(images)
        projected_features = self.feature_projection(visual_features)
        classification_output = self.classifier(visual_features)

        if generate and text_inputs is not None:
            encoder_hidden_states = projected_features.unsqueeze(1)
            attention_mask = torch.ones_like(text_inputs)

            generated_ids = self.text_decoder.generate(
                input_ids=text_inputs,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                max_length=max_length,
                pad_token_id=self.text_decoder.config.pad_token_id,
                eos_token_id=self.text_decoder.config.eos_token_id,
                bos_token_id=self.text_decoder.config.bos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.9
            )
            return classification_output, generated_ids

        elif text_inputs is not None:
            seq_len = text_inputs.shape[1]
            repeated_features = projected_features.unsqueeze(1).repeat(1, seq_len, 1)
            logits = self.text_decoder(
                input_ids=text_inputs,
                encoder_hidden_states=repeated_features
            ).logits
            return classification_output, logits

        return classification_output, None

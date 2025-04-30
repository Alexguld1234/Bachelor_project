# model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config
import torchxrayvision as xrv

# ---------------- Encoder Options ----------------

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

class DenseNet121Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = xrv.models.DenseNet(weights="all" if pretrained else None)
        self.model.op_threshs = None  # Turn off built-in thresholds

    def forward(self, x):
        features = self.model.features(x)
        features = torch.nn.functional.relu(features, inplace=True)
        features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
        return features.view(features.size(0), -1)

class ScratchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
    
    def forward(self, x):
        x = self.conv(x)
        return torch.flatten(x, 1)

# ---------------- Decoder Options ----------------

class ScratchTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=2, num_heads=4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, encoder_hidden_states=None):
        x = self.embedding(input_ids)
        if encoder_hidden_states is not None:
            if encoder_hidden_states.dim() == 2:  # [B, D]
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # [B, 1, D]
            x = x + encoder_hidden_states  # broadcast to [B, T, D]
  # add features as bias
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits
    def generate(self, input_ids, encoder_hidden_states=None, max_length=128, **kwargs):
        """
        Greedy decoding for the scratch transformer.
        input_ids: [batch_size, seq_len] (prompt tokens)
        encoder_hidden_states: [batch_size, hidden_dim]
        """
        generated = input_ids.clone()
        batch_size = input_ids.size(0)
        device = input_ids.device

        for _ in range(max_length - input_ids.size(1)):
            logits = self.forward(generated, encoder_hidden_states=encoder_hidden_states)
            next_token_logits = logits[:, -1, :]  # get logits for last token
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)  # greedy pick
            generated = torch.cat((generated, next_token), dim=1)

        return generated


# ---------------- Full Model ----------------

class RadTexModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, vocab_size, num_classes=4, pretrained_backbone=True):
        super().__init__()

        # Select Encoder
        if encoder_name == "resnet":
            self.visual_backbone = ResNet50Backbone(pretrained=pretrained_backbone)
            visual_feature_dim = 2048
        elif encoder_name == "densenet":
            self.visual_backbone = DenseNet121Backbone(pretrained=pretrained_backbone)
            visual_feature_dim = 1024
        elif encoder_name == "scratch_encoder":
            self.visual_backbone = ScratchCNN()
            visual_feature_dim = 64
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

        # Select Decoder
        if decoder_name == "gpt2":
            config = GPT2Config.from_pretrained("gpt2")
            config.add_cross_attention = True
            config.vocab_size = vocab_size
            config.pad_token_id = config.eos_token_id
            config.bos_token_id = config.bos_token_id or config.eos_token_id
            self.text_decoder = GPT2LMHeadModel(config)
            self.text_decoder.resize_token_embeddings(vocab_size)
        elif decoder_name == "biogpt":
            self.text_decoder = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
        # Only resize if vocab size doesn't match
            if self.text_decoder.config.vocab_size != vocab_size:
                self.text_decoder.resize_token_embeddings(vocab_size)

        elif decoder_name == "scratch_decoder":
            self.text_decoder = ScratchTransformer(vocab_size)
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")
        if decoder_name == "scratch_decoder":
            decoder_input_dim = 256  # Match hidden_dim from ScratchTransformer
        else:
            decoder_input_dim = 768

        self.feature_projection = nn.Linear(visual_feature_dim, decoder_input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(visual_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, images, text_inputs=None, generate=False, generation_args=None):
        visual_features = self.visual_backbone(images)
        projected_features = self.feature_projection(visual_features)
        classification_output = self.classifier(visual_features)

        if generate and text_inputs is not None:
            encoder_hidden_states = projected_features.unsqueeze(1)
            attention_mask = torch.ones_like(text_inputs)

            if hasattr(self.text_decoder, 'generate'):  # GPT/BioGPT models
                generation_args = generation_args or {}
                base_args = {
                    "input_ids": text_inputs,
                    "attention_mask": attention_mask,
                    "max_length": generation_args.get("max_length", 128),
                    "repetition_penalty": generation_args.get("repetition_penalty", 1.2),
                    "top_k": generation_args.get("top_k", 50),
                    "top_p": generation_args.get("top_p", 0.95),
                    "do_sample": True
                }

    # Only pass encoder_hidden_states if model supports cross-attention
                if hasattr(self.text_decoder, "config") and getattr(self.text_decoder.config, "add_cross_attention", False):
                    base_args["encoder_hidden_states"] = encoder_hidden_states

                generated_ids = self.text_decoder.generate(**base_args)
            else:  # scratch decoder
                generated_ids = None  # Scratch decoder does not implement `.generate()`

            return classification_output, generated_ids

        elif text_inputs is not None:
            seq_len = text_inputs.shape[1]
            repeated_features = projected_features.unsqueeze(1).repeat(1, seq_len, 1)

            if hasattr(self.text_decoder, 'forward'):
                out = self.text_decoder(
                    input_ids=text_inputs,
                    encoder_hidden_states=repeated_features
                )
                logits = out.logits if hasattr(out, "logits") else out  # support HuggingFace or scratch
            else:
                logits = self.text_decoder(text_inputs, encoder_hidden_states=repeated_features)

            return classification_output, logits

        return classification_output, None

    def freeze_encoder(self):
        for param in self.visual_backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.visual_backbone.parameters():
            param.requires_grad = True

# ✅ Helper for pipeline use
def build_model(encoder_name, decoder_name, vocab_size, pretrained_backbone=True, freeze_encoder=False):
    model = RadTexModel(
        encoder_name=encoder_name,
        decoder_name=decoder_name,
        vocab_size=vocab_size,
        pretrained_backbone=pretrained_backbone
    )
    if freeze_encoder:
        model.freeze_encoder()
    return model

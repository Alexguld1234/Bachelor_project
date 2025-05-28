# radtex_model.py
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModelForCausalLM, GPT2LMHeadModel, GPT2Config
import torchxrayvision as xrv
import torchvision.models as tv_models

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

class DenseNetBackbone(nn.Module):
    def __init__(self, variant="densenet121", pretrained=True):
        super().__init__()
        if variant == "densenet121":
            net = tv_models.densenet121(weights=tv_models.DenseNet121_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1024
        elif variant == "densenet169":
            net = tv_models.densenet169(weights=tv_models.DenseNet169_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1664
        elif variant == "densenet201":
            net = tv_models.densenet201(weights=tv_models.DenseNet201_Weights.DEFAULT if pretrained else None)
            self.feature_dim = 1920
        elif variant == "densenet264":
            raise NotImplementedError("DenseNet-264 is not available in torchvision.")
        else:
            raise ValueError(f"Unknown DenseNet variant: {variant}")

        # Adapt first conv layer to 1 channel
        conv0 = net.features.conv0
        self.conv0 = nn.Conv2d(1, conv0.out_channels, kernel_size=conv0.kernel_size,
                               stride=conv0.stride, padding=conv0.padding, bias=False)
        with torch.no_grad():
            self.conv0.weight = nn.Parameter(conv0.weight.mean(dim=1, keepdim=True))

        self.features = net.features
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv0(x)
        x = self.features.relu0(self.features.norm0(x))
        x = self.features.denseblock1(x)
        x = self.features.transition1(x)
        x = self.features.denseblock2(x)
        x = self.features.transition2(x)
        x = self.features.denseblock3(x)
        x = self.features.transition3(x)
        x = self.features.denseblock4(x)
        x = self.features.norm5(x)
        x = self.pooling(x)
        return x.view(x.size(0), -1)

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
            if encoder_hidden_states.dim() == 2:
                encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
            x = x + encoder_hidden_states
        x = self.transformer(x)
        logits = self.fc_out(x)
        return logits

    def generate(self, input_ids, encoder_hidden_states=None, max_length=128, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_length - input_ids.size(1)):
            logits = self.forward(generated, encoder_hidden_states=encoder_hidden_states)
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat((generated, next_token), dim=1)
        return generated

# ---------------- Full Model ----------------

class RadTexModel(nn.Module):
    def __init__(self, encoder_name, decoder_name, vocab_size, num_classes=4, pretrained_backbone=True):
        super().__init__()

        if encoder_name == "resnet":
            self.visual_backbone = ResNet50Backbone(pretrained=pretrained_backbone)
            visual_feature_dim = 2048
        elif encoder_name.startswith("densenet"):
            self.visual_backbone = DenseNetBackbone(variant=encoder_name, pretrained=pretrained_backbone)
            visual_feature_dim = self.visual_backbone.feature_dim
        elif encoder_name == "scratch_encoder":
            self.visual_backbone = ScratchCNN()
            visual_feature_dim = 64
        else:
            raise ValueError(f"Unknown encoder: {encoder_name}")

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
            if self.text_decoder.config.vocab_size != vocab_size:
                self.text_decoder.resize_token_embeddings(vocab_size)
        elif decoder_name == "scratch_decoder":
            self.text_decoder = ScratchTransformer(vocab_size)
        else:
            raise ValueError(f"Unknown decoder: {decoder_name}")

        decoder_input_dim = 256 if decoder_name == "scratch_decoder" else 768
        self.feature_projection = nn.Linear(visual_feature_dim, decoder_input_dim)

        self.classifier = nn.Sequential(
            nn.Linear(visual_feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, text_inputs=None, generate=False, generation_args=None):
        visual_features = self.visual_backbone(images)
        projected_features = self.feature_projection(visual_features)
        classification_output = self.classifier(visual_features)

        if generate and text_inputs is not None:
            encoder_hidden_states = projected_features.unsqueeze(1)
            attention_mask = torch.ones_like(text_inputs)
            if hasattr(self.text_decoder, 'generate'):
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
                if hasattr(self.text_decoder, "config") and getattr(self.text_decoder.config, "add_cross_attention", False):
                    base_args["encoder_hidden_states"] = encoder_hidden_states
                generated_ids = self.text_decoder.generate(**base_args)
            else:
                generated_ids = None
            return classification_output, generated_ids

        elif text_inputs is not None:
            repeated_features = projected_features.unsqueeze(1).repeat(1, text_inputs.shape[1], 1)
            out = self.text_decoder(input_ids=text_inputs, encoder_hidden_states=repeated_features)
            logits = out.logits if hasattr(out, "logits") else out
            return classification_output, logits

        return classification_output, None

    def freeze_encoder(self):
        for param in self.visual_backbone.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.visual_backbone.parameters():
            param.requires_grad = True

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

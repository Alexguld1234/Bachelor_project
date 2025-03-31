# src/train.py

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AdamW
import logging

logger = logging.getLogger(__name__)

# ✅ Load tokenizer with pad token
tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
vocab_size = tokenizer.vocab_size

def train_model(model, train_loader, val_loader, cfg, run_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    classification_criterion = nn.BCELoss()
    generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=1e-5)

    for epoch in range(cfg.train.epochs):
        model.train()
        total_class_loss, total_text_loss = 0.0, 0.0

        for images, reports, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            tokenized = tokenizer(
                reports,
                padding=True,
                truncation=True,
                max_length=cfg.run.max_length,
                return_tensors="pt"
            )
            text_inputs = tokenized["input_ids"].to(device)

            # ✅ Shift inputs for language modeling
            input_ids = text_inputs[:, :-1]
            target_ids = text_inputs[:, 1:]

            optimizer.zero_grad()
            class_output, text_output = model(images, text_inputs=input_ids)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, vocab_size), target_ids.reshape(-1))
            total_loss = class_loss + text_loss

            total_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

        logger.info(
            f"Epoch [{epoch+1}/{cfg.train.epochs}] | "
            f"Class Loss: {total_class_loss / len(train_loader):.4f}, "
            f"Text Loss: {total_text_loss / len(train_loader):.4f}"
        )

    if cfg.run.save_model:
        model_path = run_dir / "model.pth"
        torch.save(model.state_dict(), model_path)
        logger.info(f"✅ Saved model to {model_path}")

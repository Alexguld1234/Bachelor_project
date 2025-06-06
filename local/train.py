# train.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import Adam
from tqdm import tqdm
import pandas as pd
from pathlib import Path

from data import get_dataloader
from radtex_model import RadTexModel
def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "left"
    return tokenizer

def train(model, train_loader, val_loader, tokenizer, epochs=1, lr=2e-5, save_path="radtex_model.pth", device="cuda", 
          only_classification=False, only_text_generation=False, generation_args=None):
    model = model.to(device)
    vocab_size = tokenizer.vocab_size

    classification_criterion = nn.CrossEntropyLoss()
    generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    
    epoch_results = []
    phase = "classification" if only_classification else "text_generation"

    for epoch in range(epochs):
        

        print(f"\n🔥 Epoch {epoch+1}/{epochs}")
        model.train()
        total_class_loss, total_text_loss = 0.0, 0.0

        for images, reports, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).long()

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(device)

            optimizer.zero_grad()

            # Forward pass
            class_output, text_output = model(
                images,
                text_inputs=text_inputs,
                generate=False,
                generation_args=generation_args or {}
            )

            total_loss = 0.0

            if not only_text_generation:
                class_loss = classification_criterion(class_output, labels)
                total_loss += class_loss
                total_class_loss += class_loss.item()

            if not only_classification and text_output is not None:
                text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))
                total_loss += text_loss
                total_text_loss += text_loss.item()

            total_loss.backward()
            optimizer.step()

        print(f"🧪 Epoch [{epoch+1}/{epochs}] | Class Loss: {total_class_loss / len(train_loader):.4f}, Text Loss: {total_text_loss / len(train_loader):.4f}")

        validate(model, val_loader, tokenizer, vocab_size, classification_criterion, generation_criterion, device, only_classification, only_text_generation)
        epoch_results.append({
        "epoch": epoch + 1,
        "class_loss": total_class_loss / len(train_loader),
        "text_loss": total_text_loss / len(train_loader)
        })
    if save_path:
        save_dir = Path(save_path).parent
        (save_dir / "evaluations").mkdir(parents=True, exist_ok=True)
    if phase == "classification":
        filename = "classification_epoch_metrics.csv"
    else:
        filename = "text_gen_epoch_metrics.csv"
    pd.DataFrame(epoch_results).to_csv(save_dir / "evaluations" / filename, index=False)

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")
    return model
def validate(model, val_loader, tokenizer, vocab_size, classification_criterion, generation_criterion, device, only_classification=False, only_text_generation=False):
    model.eval()
    correct, total, base_correct = 0, 0, 0
    total_class_loss, total_text_loss = 0.0, 0.0

    with torch.no_grad():
        for images, reports, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(device)

            class_output, text_output = model(
                images,
                text_inputs=text_inputs,
                generate=False
            )

            if not only_text_generation:
                class_loss = classification_criterion(class_output, labels)
                total_class_loss += class_loss.item()

            if not only_classification and text_output is not None:
                text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))
                total_text_loss += text_loss.item()

            predicted = class_output.argmax(dim=1)
            print(f"Predicted: {predicted}, Labels: {labels}")
            correct += (predicted == labels).sum().item()
            base_correct += ((torch.tensor([2] * labels.size(0)).to(device)) == labels).sum().item()
            total += labels.size(0)


    base_acc = 100 * base_correct / total 
    print(f"📊 Validation — Base Accuracy: {base_acc:.2f}%")
    acc = 100 * correct / total
    print(f"🔍 Validation — Class Loss: {total_class_loss / len(val_loader):.4f}, Text Loss: {total_text_loss / len(val_loader):.4f}, Accuracy: {acc:.2f}%")

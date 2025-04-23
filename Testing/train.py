import torch
import torch.nn as nn
from transformers import AdamW, AutoTokenizer
from data import get_dataloader
from model import RadTexModel
from tqdm import tqdm
import os

def train_model(
    output_dir,
    epochs=1,
    batch_size=16,
    learning_rate=2e-5,
    checkpoint_path=None,
    save_checkpoint=True,
    device=None,
    max_length=128
):
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = tokenizer.vocab_size

    train_loader = get_dataloader(mode="train", batch_size=batch_size, shuffle=True)
    val_loader = get_dataloader(mode="val", batch_size=batch_size, shuffle=False)

    model = RadTexModel(vocab_size=vocab_size).to(device)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"🔁 Loading checkpoint: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    classification_criterion = nn.BCELoss()
    generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    for epoch in range(epochs):
        print(f"\n🔄 Epoch {epoch+1}/{epochs}")
        model.train()
        total_class_loss, total_text_loss = 0.0, 0.0

        for images, reports, labels in tqdm(train_loader):
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(device)

            optimizer.zero_grad()
            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))
            total_loss = class_loss + text_loss

            total_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

        print(f"✅ Training Loss - Class: {total_class_loss / len(train_loader):.4f}, Text: {total_text_loss / len(train_loader):.4f}")

        # Validate after each epoch
        validate(model, val_loader, tokenizer, device, vocab_size, max_length)

        # Save checkpoint
        if save_checkpoint:
            checkpoint_file = os.path.join(output_dir, f"checkpoint_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_file)
            print(f"💾 Saved checkpoint to: {checkpoint_file}")

    return model

def validate(model, val_loader, tokenizer, device, vocab_size, max_length):
    model.eval()
    correct, total = 0, 0
    total_class_loss, total_text_loss = 0.0, 0.0

    classification_criterion = nn.BCELoss()
    generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    with torch.no_grad():
        for images, reports, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(device)

            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

            predicted = (class_output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"📊 Validation - Class Loss: {total_class_loss / len(val_loader):.4f}, Text Loss: {total_text_loss / len(val_loader):.4f}, Accuracy: {acc:.2f}%")


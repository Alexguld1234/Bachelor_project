import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import Adam
from tqdm import tqdm

from data import get_dataloader
from model import RadTexModel

def get_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    tokenizer.padding_side = "left"
    return tokenizer

def train(model, train_loader, val_loader, tokenizer, epochs=1, lr=2e-5, save_path="radtex_model.pth", device="cuda"):
    model = model.to(device)
    vocab_size = tokenizer.vocab_size

    classification_criterion = nn.CrossEntropyLoss()
    generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

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
            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))
            total_loss = class_loss + text_loss

            total_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

        print(f"🧪 Epoch [{epoch+1}/{epochs}] | Class Loss: {total_class_loss / len(train_loader):.4f}, Text Loss: {total_text_loss / len(train_loader):.4f}")
        validate(model, val_loader, tokenizer, vocab_size, classification_criterion, generation_criterion, device)

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved at {save_path}")

def validate(model, val_loader, tokenizer, vocab_size, classification_criterion, generation_criterion, device):
    model.eval()
    correct, total = 0, 0
    total_class_loss, total_text_loss = 0.0, 0.0

    with torch.no_grad():
        for images, reports, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).long()

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(device)

            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, vocab_size), text_inputs.view(-1))

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

            predicted = class_output.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"🔍 Validation — Class Loss: {total_class_loss / len(val_loader):.4f}, Text Loss: {total_text_loss / len(val_loader):.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = get_tokenizer()

    train_loader = get_dataloader(mode="train", batch_size=16)
    val_loader = get_dataloader(mode="val", batch_size=16)

    model = RadTexModel(vocab_size=tokenizer.vocab_size, num_classes=4)
    train(model, train_loader, val_loader, tokenizer, epochs=1, device=DEVICE)

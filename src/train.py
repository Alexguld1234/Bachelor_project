import torch
import torch.nn as nn
from transformers import AutoTokenizer
from torch.optim import Adam


from data import get_dataloader
from model import RadTexModel
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
tokenizer.padding_side = "left"

BATCH_SIZE = 16
EPOCHS = 1                
LEARNING_RATE = 2e-10
VOCAB_SIZE = tokenizer.vocab_size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

train_loader = get_dataloader(mode="train", batch_size=BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(mode="val", batch_size=BATCH_SIZE, shuffle=False)

model = RadTexModel(vocab_size=VOCAB_SIZE).to(DEVICE)

classification_criterion = nn.BCELoss()
generation_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)


def train():
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        model.train()
        total_class_loss, total_text_loss = 0.0, 0.0

        for images, reports, labels in tqdm(train_loader):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(DEVICE)

            optimizer.zero_grad()
            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, VOCAB_SIZE), text_inputs.view(-1))
            total_loss = class_loss + text_loss

            total_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Class Loss: {total_class_loss / len(train_loader):.4f}, Text Loss: {total_text_loss / len(train_loader):.4f}")
        validate()

    torch.save(model.state_dict(), "radtex_model_epoch1.pth")
    print("✅ Model Saved as radtex_model.pth")

def validate():
    model.eval()
    correct, total = 0, 0
    total_class_loss, total_text_loss = 0.0, 0.0

    with torch.no_grad():
        for images, reports, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            tokenized = tokenizer(reports, padding=True, truncation=True, max_length=128, return_tensors="pt")
            text_inputs = tokenized["input_ids"].to(DEVICE)

            class_output, text_output = model(images, text_inputs=text_inputs)

            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, VOCAB_SIZE), text_inputs.view(-1))

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

            predicted = (class_output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    print(f"✅ Validation - Class Loss: {total_class_loss / len(val_loader):.4f}, Text Loss: {total_text_loss / len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AdamW
from data import get_dataloader
from model import RadTexModel

# ✅ Training Configuration
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 2e-4
VOCAB_SIZE = 30522  # GPT-2 default vocab size
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ Load Data
train_loader = get_dataloader(mode="train", batch_size=BATCH_SIZE, shuffle=True)
val_loader = get_dataloader(mode="val", batch_size=BATCH_SIZE, shuffle=False)

# ✅ Load Model
model = RadTexModel(vocab_size=VOCAB_SIZE).to(DEVICE)

# ✅ Define Losses
classification_criterion = nn.BCELoss()  # Binary classification loss
generation_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding tokens in text

# ✅ Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

def train():
    """
    Train the model for multiple epochs.
    """
    for epoch in range(EPOCHS):
        model.train()
        total_class_loss, total_text_loss = 0.0, 0.0

        for images, reports, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)  # (batch, 1)
            
            # Convert text reports to tokenized tensors (TODO: Use tokenizer)
            text_inputs = torch.randint(0, VOCAB_SIZE, (images.shape[0], 30)).to(DEVICE)  # Dummy tokenized input

            optimizer.zero_grad()

            # Forward Pass
            class_output, text_output = model(images, text_inputs)

            # Compute Loss
            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, VOCAB_SIZE), text_inputs.view(-1))  # Flattened for CE Loss

            total_loss = class_loss + text_loss
            total_loss.backward()
            optimizer.step()

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

        avg_class_loss = total_class_loss / len(train_loader)
        avg_text_loss = total_text_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Class Loss: {avg_class_loss:.4f}, Text Loss: {avg_text_loss:.4f}")

        # ✅ Validation Step
        validate()

    # ✅ Save Model
    torch.save(model.state_dict(), "radtex_model.pth")
    print("✅ Model Saved as radtex_model.pth")


def validate():
    """
    Validate the model on the validation set.
    """
    model.eval()
    correct, total = 0, 0
    total_class_loss, total_text_loss = 0.0, 0.0

    with torch.no_grad():
        for images, reports, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).float().unsqueeze(1)

            text_inputs = torch.randint(0, VOCAB_SIZE, (images.shape[0], 30)).to(DEVICE)  # Dummy tokenized input

            # Forward pass
            class_output, text_output = model(images, text_inputs)

            # Compute Loss
            class_loss = classification_criterion(class_output, labels)
            text_loss = generation_criterion(text_output.view(-1, VOCAB_SIZE), text_inputs.view(-1))

            total_class_loss += class_loss.item()
            total_text_loss += text_loss.item()

            # Compute accuracy for classification
            predicted = (class_output > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    avg_class_loss = total_class_loss / len(val_loader)
    avg_text_loss = total_text_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f"✅ Validation - Class Loss: {avg_class_loss:.4f}, Text Loss: {avg_text_loss:.4f}, Accuracy: {accuracy:.2f}%")


if __name__ == "__main__":
    train()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from data import get_dataloader
from model import ChestXrayReportGenerator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Hyperparameters
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
VOCAB_SIZE = 5000  # Placeholder vocab size
SAVE_PATH = Path("models")  # Model checkpoint directory
SAVE_PATH.mkdir(parents=True, exist_ok=True)

# Load Data
train_loader = get_dataloader("data/subset_pneumonia_30.csv", "data/JPG_AP", "data/Reports", mode="train", batch_size=BATCH_SIZE)
val_loader = get_dataloader("data/subset_pneumonia_30.csv", "data/JPG_AP", "data/Reports", mode="val", batch_size=BATCH_SIZE)

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChestXrayReportGenerator(VOCAB_SIZE).to(device)

# Loss functions
classification_loss_fn = nn.BCELoss()  # Binary Cross Entropy for pneumonia prediction
caption_loss_fn = nn.CrossEntropyLoss()  # Cross-Entropy for text generation

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train():
    logging.info("Starting training...")
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for images, reports, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float()

            # Convert reports to tokenized format (Placeholder)
            tokenized_reports = torch.randint(0, VOCAB_SIZE, (reports.shape[0], 20)).to(device)  # Simulated tokenization

            # Forward pass
            class_pred, report_pred = model(images, tokenized_reports)

            # Compute Loss
            class_loss = classification_loss_fn(class_pred.squeeze(), labels)
            report_loss = caption_loss_fn(report_pred.view(-1, VOCAB_SIZE), tokenized_reports.view(-1))

            loss = class_loss + report_loss
            total_train_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation Phase
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for images, reports, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float()
                tokenized_reports = torch.randint(0, VOCAB_SIZE, (reports.shape[0], 20)).to(device)

                class_pred, report_pred = model(images, tokenized_reports)

                class_loss = classification_loss_fn(class_pred.squeeze(), labels)
                report_loss = caption_loss_fn(report_pred.view(-1, VOCAB_SIZE), tokenized_reports.view(-1))
                loss = class_loss + report_loss
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        logging.info(f"Epoch [{epoch+1}/{EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH / "best_model.pth")
            logging.info(f"Saved best model at epoch {epoch+1}")

if __name__ == "__main__":
    train()

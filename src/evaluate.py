import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from model import ChestXrayReportGenerator
from data import get_dataloader
import evaluate  # HuggingFace Evaluate library for BLEU & ROUGE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load evaluation metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 5000  # Placeholder for vocab size
model = ChestXrayReportGenerator(VOCAB_SIZE).to(device)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# Load test data
test_loader = get_dataloader("data/subset_pneumonia_30.csv", "data/JPG_AP", "data/Reports", mode="test", batch_size=8)

# Define loss functions
classification_loss_fn = nn.BCELoss()
caption_loss_fn = nn.CrossEntropyLoss()

def evaluate_model():
    logging.info("Evaluating model...")

    all_labels, all_preds = [], []
    all_generated_reports, all_reference_reports = [], []

    total_test_loss = 0

    with torch.no_grad():
        for images, reports, labels in test_loader:
            images, labels = images.to(device), labels.to(device).float()

            # Convert reports to tokenized format (Placeholder)
            tokenized_reports = torch.randint(0, VOCAB_SIZE, (reports.shape[0], 20)).to(device)

            # Forward pass
            class_pred, report_pred = model(images, tokenized_reports)

            # Compute Loss
            class_loss = classification_loss_fn(class_pred.squeeze(), labels)
            report_loss = caption_loss_fn(report_pred.view(-1, VOCAB_SIZE), tokenized_reports.view(-1))
            loss = class_loss + report_loss
            total_test_loss += loss.item()

            # Store classification results
            class_preds = (class_pred.squeeze() > 0.5).cpu().numpy()
            all_preds.extend(class_preds)
            all_labels.extend(labels.cpu().numpy())

            # Store text predictions (placeholder)
            generated_reports = ["generated report text" for _ in reports]  # Replace with actual generated text
            all_generated_reports.extend(generated_reports)
            all_reference_reports.extend(reports)

    # Compute classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    # Compute text generation metrics
    bleu_score = bleu.compute(predictions=all_generated_reports, references=all_reference_reports)
    rouge_score = rouge.compute(predictions=all_generated_reports, references=all_reference_reports)

    logging.info(f"Test Loss: {total_test_loss / len(test_loader):.4f}")
    logging.info(f"Classification - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    logging.info(f"Report Generation - BLEU: {bleu_score['bleu']:.4f}, ROUGE-L: {rouge_score['rougeL']:.4f}")

if __name__ == "__main__":
    evaluate_model()

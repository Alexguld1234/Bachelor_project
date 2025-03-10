import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from model import ChestXrayReportGenerator
from data import get_dataloader
from pathlib import Path
from PIL import Image
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load best model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 5000
model = ChestXrayReportGenerator(VOCAB_SIZE).to(device)
model.load_state_dict(torch.load("models/best_model.pth"))
model.eval()

# Load test data
test_loader = get_dataloader("data/subset_pneumonia_30.csv", "data/JPG_AP", "data/Reports", mode="test", batch_size=1)

def plot_sample_predictions(num_samples=5):
    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 4))

    with torch.no_grad():
        for i, (image, report, label) in enumerate(test_loader):
            if i >= num_samples:
                break
            image = image.to(device)
            label = label.item()
            
            # Forward pass
            class_pred, _ = model(image, torch.randint(0, VOCAB_SIZE, (1, 20)).to(device))
            predicted_label = "Pneumonia" if class_pred.item() > 0.5 else "No Pneumonia"
            
            # Load original image
            img = np.array(image.cpu().squeeze(0).permute(1, 2, 0))

            # Plot image and report
            axes[i, 0].imshow(img)
            axes[i, 0].axis("off")
            axes[i, 0].set_title(f"Actual: {'Pneumonia' if label else 'No Pneumonia'}\nPredicted: {predicted_label}")

            axes[i, 1].text(0.1, 0.5, f"Generated Report:\n{report}", fontsize=12, wrap=True)
            axes[i, 1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_roc_curve(labels, predictions):
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (area = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(labels, predictions):
    precision, recall, _ = precision_recall_curve(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="red", lw=2, label="Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == "__main__":
    plot_sample_predictions()

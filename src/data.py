import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

# ✅ Set Project Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_FILE = DATA_DIR / "subset_pneumonia_30.csv"
IMG_DIR = DATA_DIR / "JPG_AP"
REPORTS_DIR = DATA_DIR / "Reports"

def set_seed(seed=42):
    """Ensures reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ChestXrayDataset(Dataset):
    def __init__(self, mode="train", split=(0.8, 0.1, 0.1), csv_file=CSV_FILE, transform=None):
        set_seed(42)
        print(f"📂 Loading dataset from: {csv_file}")
        self.data = pd.read_csv(csv_file)
        self.mode = mode
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # ✅ Force Grayscale
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.data["txt_path"] = self.data["study_id"].apply(lambda x: REPORTS_DIR / f"s{x}.txt")
        self.data["jpg_path"] = self.data["dicom_id"].apply(lambda x: IMG_DIR / f"{x}.jpg")

        dataset_size = len(self.data)
        train_size = int(split[0] * dataset_size)
        val_size = int(split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(
            self.data.to_dict(orient="records"), [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
        )

        if self.mode == "train":
            self.current_data = self.train_data
        elif self.mode == "val":
            self.current_data = self.val_data
        elif self.mode == "test":
            self.current_data = self.test_data
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

        print(f"✅ Dataset Loaded: {self.mode} set with {len(self.current_data)} samples.")

    def __len__(self):
        return len(self.current_data)

    def __getitem__(self, idx):
        sample = self.current_data[idx]

        # Load image
        img_path = sample["jpg_path"]
        image = Image.open(img_path).convert("L")  # ✅ Force Grayscale
        image = self.transform(image)  # Apply transforms

        # Load text report
        txt_path = sample["txt_path"]
        try:
            with open(txt_path, "r", encoding="utf-8") as file:
                text_report = file.read()
        except FileNotFoundError:
            text_report = ""

        label = torch.tensor(sample.get("label", 0), dtype=torch.float)

        return image, text_report, label

def get_dataloader(mode="train", batch_size=32, shuffle=True, transform=None):
    """
    Returns a DataLoader for the specified dataset split.
    """
    dataset = ChestXrayDataset(mode=mode, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    # ✅ Test DataLoader
    dataloader = get_dataloader("train", batch_size=4)
    for images, reports, labels in dataloader:
        print(f"Batch - Images: {images.shape}, Reports: {len(reports)}, Labels: {labels.shape}")
        break

import os
import pandas as pd
import torch
import logging
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import argparse

# Setup logging - MLOps best practice for tracking issues
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, img_dir, reports_dir, transform=None, mode="train", split=(0.8, 0.1, 0.1)):
        """
        Args:
            csv_file (str): Path to subset_pneumonia_30.csv.
            img_dir (str): Directory with JPG_AP images.
            reports_dir (str): Directory with free-text reports.
            transform (callable, optional): Optional transform to apply to images.
            mode (str): "train", "val", or "test".
            split (tuple): (train, val, test) fractions.
        """
        self.data = pd.read_csv(csv_file)
        self.img_dir = Path(img_dir)
        self.reports_dir = Path(reports_dir)
        self.transform = transform
        self.mode = mode

        # Splitting dataset
        dataset_size = len(self.data)
        train_size = int(split[0] * dataset_size)
        val_size = int(split[1] * dataset_size)
        test_size = dataset_size - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(self.data.to_dict(orient="records"), [train_size, val_size, test_size])

        if self.mode == "train":
            self.data = self.train_data
        elif self.mode == "val":
            self.data = self.val_data
        elif self.mode == "test":
            self.data = self.test_data
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Load Image
        img_path = self.img_dir / f"{sample['dicom_id']}.jpg"
        image = Image.open(img_path).convert("RGB")  # Ensure it's RGB

        # Load Report
        report_path = self.reports_dir / f"s{sample['study_id']}.txt"
        with open(report_path, "r", encoding="utf-8") as f:
            report_text = f.read().strip()

        # Get label
        label = sample["Pneumonia"]

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return image, report_text, label


def get_dataloader(csv_path, img_dir, reports_dir, mode="train", batch_size=8, split=(0.8, 0.1, 0.1), num_workers=2):
    """
    Returns a PyTorch DataLoader for the dataset.
    """
    dataset = ChestXrayDataset(csv_path, img_dir, reports_dir, mode=mode, split=split)
    shuffle = mode == "train"
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


if __name__ == "__main__":
    # CLI Argument Parsing - MLOps best practice for modularity
    parser = argparse.ArgumentParser(description="Chest X-ray Data Loader")
    parser.add_argument("--csv_path", type=str, default="data/subset_pneumonia_30.csv", help="Path to the CSV file")
    parser.add_argument("--img_dir", type=str, default="data/JPG_AP", help="Path to the images directory")
    parser.add_argument("--reports_dir", type=str, default="data/Reports", help="Path to the reports directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for DataLoader")
    parser.add_argument("--mode", type=str, choices=["train", "val", "test"], default="train", help="Dataset mode")
    args = parser.parse_args()

    dataloader = get_dataloader(args.csv_path, args.img_dir, args.reports_dir, mode=args.mode, batch_size=args.batch_size)

    # Example loop through the dataset
    for images, reports, labels in dataloader:
        logging.info(f"Batch Size: {len(images)}")
        logging.info(f"First Image Shape: {images[0].size}")
        logging.info(f"First Report: {reports[0][:100]}...")  # Show only first 100 chars
        logging.info(f"Labels: {labels}")
        break

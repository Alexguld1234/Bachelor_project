import argparse
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import random
from pathlib import Path
from PIL import Image, ImageFile
import torchvision.transforms as transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True


CSV_FILE = "D:\Bachelor_project/local/All_AP_pic_w_urls_50000.csv"



def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChestXrayDataset(Dataset):
    def __init__(self,
                mode: str = "train",
                split: tuple = (0.8, 0.1, 0.1),
                csv_file: Path = CSV_FILE,
                transform = None,
                num_samples: int = None):
        
        set_seed(42)
        print(f"Loading dataset from {csv_file}. Amount: {num_samples}. Mode: {mode}.")
        self.data = pd.read_csv(csv_file)

        if num_samples is not None:
            self.data = self.data.sample(n=num_samples, random_state=42).reset_index(drop=True)
            print(f"Sampled {num_samples} rows from the dataset.")
        

        self.mode = mode
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        self.data["txt_path"] = self.data["txt_urls"].apply(Path)
        self.data["jpg_path"] = self.data["urls"].apply(Path)

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
        
        print(f"✅ Loaded {mode} split: {len(self.current_data)} samples")

    def __len__(self):
        return len(self.current_data)
    
    def __getitem__(self, idx):
        sample = self.current_data[idx]


        # Load image
        img_path = sample["jpg_path"]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)

        txt_path = sample["txt_path"]
        try:
            with open(txt_path, "r") as f:
                text_report = f.read()
        except FileNotFoundError:
            print(f"File not found: {txt_path}")
            text_report = ""

        label = torch.tensor(sample.get("pneumonia_label", 3), dtype=torch.long)

        return image, text_report, label
    
def get_dataloader(mode: str = "train",
                   batch_size: int = 32, 
                   shuffle: bool = True,
                   num_samples: int = None,
                   transform=None):
    dataset = ChestXrayDataset(mode=mode, 
                               num_samples=num_samples, 
                               transform=transform)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle)

def main():
    parser = argparse.ArgumentParser(description="Chest X-ray Dataset Loader")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Mode of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to load from the dataset")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the dataset")
    parser.add_argument("--transform", type=str, default=None, help="Transformations to apply to the images")
    args = parser.parse_args()

    transform = None

    dataloader = get_dataloader(
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        transform=transform,
        num_samples=args.num_samples
    )
    # Inspect one batch
    for images, reports, labels in dataloader:
        print(f"Batch → Images: {images.shape}, Reports: {len(reports)}, Labels: {labels.shape}")
        break
    

if __name__ == "__main__":
    main()
      
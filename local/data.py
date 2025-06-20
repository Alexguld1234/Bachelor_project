import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split


ImageFile.LOAD_TRUNCATED_IMAGES = True
csv_file="local/Final_AP_Missing_Removed_12519.csv"


def set_seed(seed: int = 42) -> None:
    """Ensure reproducible results across Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ChestXrayDataset(Dataset):
    def __init__(self,
                mode: str = "train",
                split: tuple = (0.8, 0.1, 0.1),
                setup: str = "local",
                img_size: tuple = (224, 224),
                num_samples: int = None,
                csv_file: Path = Path("local/Final_AP_url_label_50000.csv")):   # NEW):
        set_seed(42)
        self.data = pd.read_csv(csv_file)

        if num_samples is not None:
            self.data = self.data.sample(n=num_samples, random_state=42).reset_index(drop=True)
        
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        if setup == "local":
            self.data["txt_path"] = self.data["local_txt_urls"].apply(Path)
            self.data["jpg_path"] = self.data["local_urls"].apply(Path)
        else:
            self.data["txt_path"] = self.data["hpc_txt_urls"].apply(Path)
            self.data["jpg_path"] = self.data["hpc_urls"].apply(Path)

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
        

    def __len__(self):
        return len(self.current_data)
    
    def __getitem__(self, idx):
        sample = self.current_data[idx]


        # Load image
        img_path = sample["jpg_path"]
        image = Image.open(img_path)
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
                   img_size: tuple = (224, 224),
                   setup: str = "local",                                 # NEW
                   csv_file: Path = Path("local/Final_AP_url_label_50000.csv") # NEW
                   ):
    dataset = ChestXrayDataset(mode=mode,
                               num_samples=num_samples,
                               img_size=img_size,
                               setup=setup,
                               csv_file=csv_file)                        # pass them on
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle)
def main():
    parser = argparse.ArgumentParser(description="Chest X-ray Dataset Loader")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "val", "test"], help="Mode of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for DataLoader")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to load from the dataset")
    parser.add_argument("--shuffle", action='store_true', help="Shuffle the dataset")
    parser.add_argument("--no_shuffle", dest="shuffle", action='store_false', help="Disable shuffling")
    parser.set_defaults(shuffle=True)

    parser.add_argument("--setup", choices=["local", "hpc"], default="local")
    parser.add_argument("--csv_file", type=str, default="local/Final_AP_url_label_50000.csv")
    parser.add_argument("--img_size", type=int, default=224, help="Square side length")
    args = parser.parse_args()

    dataloader = get_dataloader(
        mode=args.mode,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_samples=args.num_samples,
        img_size=(args.img_size, args.img_size),
        setup=args.setup,
        csv_file=Path(args.csv_file),
)
    
    for images, text_reports, labels in dataloader:
        print(f"Images batch shape: {images.size()}")
        print(f"Text reports batch size: {len(text_reports)}")
        print(f"Labels batch shape: {labels.size()}")
        break

if __name__ == "__main__":
    main()
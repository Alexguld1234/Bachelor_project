import argparse
import os
import sys
import yaml
import shutil
import time
import torch
import platform
from pathlib import Path
from datetime import datetime

from local.radtex_model import build_model
from data import get_dataloader
from train import train, get_tokenizer
from evaluate import evaluate
from visualize import visualize_predictions

def log_system_info(save_dir):
    info = {
        "Python version": sys.version,
        "Platform": platform.platform(),
        "CUDA available": torch.cuda.is_available(),
        "GPU name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }
    with open(save_dir / "system_info.yaml", "w") as f:
        yaml.dump(info, f)

def save_config(args, save_dir):
    config_path = save_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(vars(args), f)

def main():
    parser = argparse.ArgumentParser(description="Full Chest X-ray Pipeline")

    # Experiment setup
    parser.add_argument("--name", type=str, required=True, help="Run name")
    parser.add_argument("--save_path", type=str, default="./runs", help="Base directory to save outputs")

    # Training params
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_datapoints", type=int, default=None, help="Subset of dataset")

    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")

    args = parser.parse_args()

    # Create output folder
    run_time = datetime.now().strftime("%d-%m-%y--%H.%M")
    run_dir = Path(args.save_path) / f"{args.name}_{run_time}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config and system info
    save_config(args, run_dir)
    log_system_info(run_dir)

    # Start timer
    start_time = time.time()

    # Tokenizer & model
    tokenizer = get_tokenizer()
    model = build_model(vocab_size=tokenizer.vocab_size, pretrained_backbone=True)

    # Resume training
    checkpoint_path = run_dir / "checkpoint.pth"
    if args.resume and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
        print("✅ Resumed from checkpoint.")

    # Dataloaders
    train_loader = get_dataloader(mode="train", batch_size=args.batch_size, num_samples=args.num_datapoints)
    val_loader = get_dataloader(mode="val", batch_size=args.batch_size, num_samples=args.num_datapoints)

    # Train
    train(model=model,
          train_loader=train_loader,
          val_loader=val_loader,
          tokenizer=tokenizer,
          epochs=args.epochs,
          lr=args.learning_rate,
          save_path=str(checkpoint_path),
          device="cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate
    sys.stdout = open(run_dir / "log.txt", "w")
    evaluate(model_path=str(checkpoint_path),
             batch_size=args.batch_size,
             device="cuda" if torch.cuda.is_available() else "cpu")

    # Visualize
    sys.stdout = sys.__stdout__
    visualize_predictions(model_path=str(checkpoint_path), batch_size=args.batch_size)

    # Save runtime
    total_time = time.time() - start_time
    with open(run_dir / "runtime.txt", "w") as f:
        f.write(f"Total time (s): {total_time:.2f}\n")

    print(f"\n✅ Done! Outputs saved to: {run_dir}")

if __name__ == "__main__":
    main()

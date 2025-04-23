import argparse
import os
import json
from datetime import datetime
import torch
import logging

from train import train_model
from evaluate import evaluate_model
from visualize import show_predictions

def create_output_dir(base_name="run", base_path="./outputs"):
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_path, f"{base_name}_{run_time}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    return output_dir

def save_config(config_dict, output_dir):
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

def save_readme(config_dict, output_dir):
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Run Summary\n\n")
        f.write(f"**Timestamp:** {datetime.now().isoformat()}\n\n")
        f.write("## Config:\n")
        for k, v in config_dict.items():
            f.write(f"- **{k}**: {v}\n")

def log_setup(output_dir):
    log_path = os.path.join(output_dir, "log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Run pipeline for RadTex")
    parser.add_argument("--name", type=str, default="run", help="Run name")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def main():
    args = parse_args()

    config = {
        "name": args.name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "max_length": args.max_length,
        "resume_checkpoint": args.resume,
        "device": args.device,
    }

    output_dir = create_output_dir(args.name)
    save_config(config, output_dir)
    save_readme(config, output_dir)
    log_setup(output_dir)

    logging.info("🚀 Starting Training Run")
    logging.info(f"Configuration: {config}")
    logging.info(f"Device: {args.device}")

    try:
        model = train_model(
            output_dir=os.path.join(output_dir, "checkpoints"),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            checkpoint_path=args.resume,
            device=args.device,
            max_length=args.max_length,
        )

        model_path = os.path.join(output_dir, "checkpoints", f"checkpoint_epoch{args.epochs}.pth")
        logging.info("✅ Training complete. Starting evaluation...")

        metrics = evaluate_model(model_path=model_path, batch_size=args.batch_size, max_length=args.max_length, device=args.device)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        logging.info("📊 Evaluation complete. Generating visualizations...")
        show_predictions(model_path=model_path, batch_size=args.batch_size, device=args.device)

        logging.info("🎉 Run complete!")

    except Exception as e:
        logging.exception("💥 Run failed with an exception")
        with open(os.path.join(output_dir, "FAILED.txt"), "w") as f:
            f.write(str(e))
        raise

if __name__ == "__main__":
    main()

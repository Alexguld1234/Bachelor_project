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

from model import build_model
from data import get_dataloader
from train import train, get_tokenizer
from evaluate import evaluate
from visualize import visualize_predictions

# ────────────────── CLI ──────────────────
parser = argparse.ArgumentParser(
    description="Chest-X-ray end-to-end pipeline (train → eval → visualise)"
)

# • bookkeeping
parser.add_argument("--name", type=str, default=None, help="Run folder name")
parser.add_argument("--save_path", type=Path, default=Path("/work3/s224228/bachelor_runs"),
                    help="Root dir for all runs")
parser.add_argument("--config", type=Path, default=None,
                    help="YAML config to load (overridden by later CLI flags)")

# • data
parser.add_argument("--csv_file", type=Path, help="Path to MIMIC-CXR CSV (optional)")
parser.add_argument("--batch_size", type=int, default=32, help="Images per iteration")
parser.add_argument("--num_datapoints", type=int, default=None,
                    help="Limit dataset rows for quick tests")

# • optimisation
parser.add_argument("--learning_rate", type=float, default=2e-5, help="AdamW LR")

# • model architecture
parser.add_argument("--encoder", type=str, default="resnet",
                    choices=["resnet", "densenet", "scratch_encoder"],
                    help="Visual backbone")
parser.add_argument("--decoder", type=str, default="gpt2",
                    choices=["gpt2", "biogpt", "scratch_decoder"],
                    help="Language model")

# • training phases
parser.add_argument("--training_phases", type=str, default="classification_then_text",
                    choices=["classification_only", "classification_then_text", "text_only"])
parser.add_argument("--epochs_classification", type=int, default=50)
parser.add_argument("--epochs_text_generation", type=int, default=50)
parser.add_argument("--freeze_encoder", action="store_true",
                    help="Freeze encoder during text phase")

# • generation hyper-params
parser.add_argument("--repetition_penalty", type=float, default=1.2)
parser.add_argument("--top_k", type=int, default=50)
parser.add_argument("--top_p", type=float, default=0.95)
parser.add_argument("--max_length", type=int, default=128)

# --------------- Parse args (with optional YAML) ---------------
args_cli = parser.parse_args()

if args_cli.config is not None:
    with open(args_cli.config, "r") as f:
        loaded_cfg = yaml.safe_load(f)
    parser.set_defaults(**loaded_cfg)
    # Re-parse so CLI overrides YAML
    args = parser.parse_args()
else:
    args = args_cli

# ────────────────── House-keeping ──────────────────
start_time = time.time()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = args.name or f"{args.encoder}_{args.decoder}_{timestamp}"
run_dir = (args.save_path / run_name).resolve()
run_dir.mkdir(parents=True, exist_ok=True)

# Save final config for reproducibility
# Save final config for reproducibility ── cast PosixPath → str
cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
with open(run_dir / "config.yaml", "w") as f:
    yaml.safe_dump(cfg, f)
print(f"\n📁 Outputs will be written to: {run_dir}\n")

# Log system info
with open(run_dir / "env.txt", "w") as f:
    print("Platform:", platform.platform(), file=f)
    print("Python:", sys.version.replace("\n", " "), file=f)
    print("CUDA available:", torch.cuda.is_available(), file=f)
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0), file=f)

# ────────────────── Build tokenizer & model ──────────────────
tokenizer = get_tokenizer()

model = build_model(
    encoder_name=args.encoder,
    decoder_name=args.decoder,
    vocab_size=tokenizer.vocab_size,
    pretrained_backbone=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# ────────────────── Dataloaders ──────────────────
train_loader = get_dataloader(
    mode="train",
    batch_size=args.batch_size,
    num_samples=args.num_datapoints
)
val_loader = get_dataloader(
    mode="val",
    batch_size=args.batch_size,
    num_samples=args.num_datapoints
)
test_loader = get_dataloader(mode="test", batch_size=args.batch_size)

# ────────────────── Training phases ──────────────────
ckpt_path_cls = run_dir / "encoder_only.pth"
ckpt_path_full = run_dir / "full_model.pth"

if args.training_phases == "classification_only":
    print("\n🔵 Phase: Classification Only Training")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=args.epochs_classification,
        lr=args.learning_rate,
        save_path=str(ckpt_path_cls),
        device=device,
        only_classification=True
    )
    final_ckpt = ckpt_path_cls

elif args.training_phases == "classification_then_text":
    # Phase 1
    print("\n🔵 Phase 1: Classification Training")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=args.epochs_classification,
        lr=args.learning_rate,
        save_path=str(ckpt_path_cls),
        device=device,
        only_classification=True
    )
    # Phase 2
    print("\n🟠 Phase 2: Text Generation Training")
    if args.freeze_encoder:
        model.freeze_encoder()
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=args.epochs_text_generation,
        lr=args.learning_rate,
        save_path=str(ckpt_path_full),
        device=device,
        only_text_generation=True,
        generation_args={
            "repetition_penalty": args.repetition_penalty,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_length": args.max_length
        }
    )
    final_ckpt = ckpt_path_full

elif args.training_phases == "text_only":
    print("\n🟠 Phase: Text Generation Only (Freeze Encoder)")
    model.freeze_encoder()
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=args.epochs_text_generation,
        lr=args.learning_rate,
        save_path=str(ckpt_path_full),
        device=device,
        only_text_generation=True,
        generation_args={
            "repetition_penalty": args.repetition_penalty,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "max_length": args.max_length
        }
    )
    final_ckpt = ckpt_path_full
else:
    raise ValueError(f"Unknown training_phases: {args.training_phases}")

# ────────────────── Evaluate & Visualise ──────────────────
evaluate(model_path=str(final_ckpt), batch_size=args.batch_size, device=device)
visualize_predictions(model_path=str(final_ckpt), batch_size=args.batch_size)

# ────────────────── Runtime summary ──────────────────
total_time = time.time() - start_time
with open(run_dir / "runtime.txt", "w") as f:
    f.write(f"Total wall-clock time (s): {total_time:.2f}\n")

print(f"\n✅ Done! Runtime: {total_time/3600:.2f} h — checkpoints & logs in {run_dir}") 
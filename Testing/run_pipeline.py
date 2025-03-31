# src/run_pipeline.py

import hydra
from omegaconf import DictConfig
import torch
import logging

from data import get_dataloader
from model import RadTexModel
from train import train_model
from evaluate import evaluate_model
from visualize import generate_visualizations
from utils.io_utils import create_run_dir, save_config, save_model, save_metrics

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="default")
def main(cfg: DictConfig):
    logger.info("Starting pipeline...")

    # Create run folder
    run_dir = create_run_dir()
    logger.info(f"Created run folder at: {run_dir}")

    # Save config
    save_config(cfg, run_dir)

    # Load data
    train_loader = get_dataloader("train", batch_size=cfg.train.batch_size)
    val_loader = get_dataloader("val", batch_size=cfg.train.batch_size)
    test_loader = get_dataloader("test", batch_size=cfg.train.batch_size)

    # Init model
    model = RadTexModel(vocab_size=30522).to("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    train_model(model, train_loader, val_loader, cfg, run_dir)

    # Evaluate
    metrics = evaluate_model(model, test_loader, cfg, run_dir)

    # Save metrics
    save_metrics(metrics, run_dir)

    # Visualize
    generate_visualizations(model, test_loader, cfg, run_dir)

if __name__ == "__main__":
    main()

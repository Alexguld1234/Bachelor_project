# src/utils/io_utils.py

import os
import json
from pathlib import Path
from datetime import datetime
import torch
import yaml
from omegaconf import OmegaConf


def create_run_dir(base_dir="runs"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    run_dir = Path(base_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "generation").mkdir(exist_ok=True)
    (run_dir / "visualizations").mkdir(exist_ok=True)
    return run_dir


def save_config(cfg, run_dir):
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(OmegaConf.to_container(cfg, resolve=True), f)

def save_metrics(metrics: dict, run_dir):
    path = Path(run_dir) / "metrics.json"
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)

def save_model(model, path):
    torch.save(model.state_dict(), path)

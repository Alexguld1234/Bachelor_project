#!/usr/bin/env bash
# -------- run_local.sh --------
set -e                         # stop on first error

# 1) activate environment
conda activate bachelor_project       # or: source venv/bin/activate

# 2) run the pipeline
python hpc/run_pipeline.py \
  --name my_local_test \
  --csv_file ~/datasets/mimic/HPC_AP_url_label_50000.csv \
  --encoder densenet \
  --decoder biogpt \
  --training_phases classification_then_text \
  --epochs_classification 20 \
  --epochs_text_generation 20 \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --num_datapoints 10000 \
  --save_path ~/runs/chestxray \
  --repetition_penalty 1.3 \
  --top_k 40 \
  --top_p 0.9

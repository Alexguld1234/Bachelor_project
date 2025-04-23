#!/bin/sh
### General options
#BSUB -q gpua100
#BSUB -J chestxray_pipeline
#BSUB -o chestxray_pipeline_%J.out
#BSUB -e chestxray_pipeline_%J.err
#BSUB -n 4
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 72:00
#BSUB -R "rusage[mem=16GB]"
#BSUB -M 16GB
#BSUB -u s224228@dtu.dk
#BSUB -B
#BSUB -N

# ✅ Activate your virtual environment
source /zhome/44/7/187366/Bachelor_project/bachelor/bin/activate

# ✅ Set CUDA device manually (optional but clean)
export CUDA_VISIBLE_DEVICES=0

# ✅ Run the full pipeline
python run_pipeline.py \
    --name chestxray_experiment \
    --epochs 1 \
    --batch_size 5 \
    --learning_rate 2e-2 \
    --num_datapoints 20 \
    --save_path /work3/s224228/bachelor_runs

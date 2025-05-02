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

# ✅ Set CUDA device manually (optional)
export CUDA_VISIBLE_DEVICES=0

# ✅ Run the full pipeline with matching parameters

python hpc/run_pipeline.py \
  --name local_resnet_gpt2_test \
  --setup hpc \
  --encoder resnet \
  --decoder gpt2 \
  --training_phases classification_then_text \
  --epochs_classification 2 \
  --epochs_text_generation 2 \
  --batch_size 8 \
  --learning_rate 3e-5 \
  --num_datapoints 10 \
  --img_size 224 \
  --save_path bachelor_runs \
  --repetition_penalty 1.2 \
  --top_k 50 \
  --top_p 0.95

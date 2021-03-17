#!/bin/bash

#SBATCH -J sg-e150-w10
#SBATCH -A jjohn273
#SBATCH -e errors.sg-e150-w10.%j.txt
#SBATCH -o output.sg-e150-w10.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem=50gb

srun poetry run python main.py \
  --iters=100 \
  --embedding-size=150 \
  --window-size=10

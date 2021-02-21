#!/bin/bash

#SBATCH -J cbow-300-w10
#SBATCH -A jjohn273
#SBATCH -e errors.cbow-300-w10.%j.txt
#SBATCH -o output.cbow-300-w10.%j.txt
#SBATCH -p longq7
#SBATCH --mem=50gb

source ~/git/Hcpcs2Vec/env/bin/activate

# module load cuda-10.1.243-gcc-8.3.0-ti55azn
# module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

srun python3 main.py iters=200 embedding_size=300 window_size=10

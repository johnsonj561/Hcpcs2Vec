#!/bin/bash

#SBATCH -J hcpcs2vec
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p gpu-bigdata7
#SBATCH --gres=gpu:1 -n1 -N1-1
#SBATCH --mem=50gb

source ~/git/Hcpcs2Vec/env/bin/activate

module load cuda-10.1.243-gcc-8.3.0-ti55azn
module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt


srun --gres=gpu:1 -n1 python3 skipgram.py \
  epochs=100 \
  embedding_size=300 \
  window_size=5 \
  x_skipgram_input=skipgrams-x-07-21-20-22-19-42.npy \
  y_skipgram_input=skipgrams-y-07-21-20-22-19-42.npy

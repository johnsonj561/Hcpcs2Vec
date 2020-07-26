#!/bin/bash

#SBATCH -J hcpcs2vec
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem=80gb
#SBATCH --exclusive
#SBATCHH -w node090


source ~/git/Hcpcs2Vec/env/bin/activate

# module load cuda-10.1.243-gcc-8.3.0-ti55azn
# module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt


srun python3 skipgram.py \
  epochs=100 \
  embedding_size=300 \
  window_size=5 \
  save_skipgrams=true

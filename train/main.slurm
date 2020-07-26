#!/bin/bash

#SBATCH -J train-sg-300
#SBATCH -A jjohn273
#SBATCH -e errors.train-sg-300.%j.txt
#SBATCH -o output.train-sg-300.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH --mem=80gb
#SBATCH --exclusive
#SBATCHH -w node090


source ~/git/Hcpcs2Vec/env/bin/activate

# module load cuda-10.1.243-gcc-8.3.0-ti55azn
# module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt

srun python3 main.py embedding_size=300

#!/bin/bash

#SBATCH -J gen-sg-20
#SBATCH -A jjohn273
#SBATCH -e errors-sg-20.%j.txt
#SBATCH -o output-sg-20.%j.txt
#SBATCH -p longq-bigdata7
#SBATCH -w node090
#SBATCH --exclusive
#SBATCH --mem=75gb

source ~/git/Hcpcs2Vec/env/bin/activate

# module load cuda-10.1.243-gcc-8.3.0-ti55azn
# module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt


srun python3 generate-skipgrams.py

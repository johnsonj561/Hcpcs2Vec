#!/bin/bash

#SBATCH -J gpu_test
#SBATCH -A jjohn273
#SBATCH -e errors.%j.txt
#SBATCH -o output.%j.txt
#SBATCH -p gpu-bigdata7
#SBATCH --gres=gpu:1 -n1 -N1-1

source ~/git/Hcpcs2Vec/env/bin/activate

module load cuda-10.1.243-gcc-8.3.0-ti55azn
module load cudnn-7.6.5.32-10.1-linux-x64-gcc-8.3.0-vldxhwt


srun --gres=gpu:1 -n1 python3 mnist_gpu.py

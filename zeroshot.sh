#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH -o test.out
#SBATCH -e test.err
#SBATCH --time 20:00:00

python eval_zeroshot.py

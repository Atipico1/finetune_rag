#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o test2.out
#SBATCH -e test2.err
#SBATCH --time 20:00:00

python preprocess.py \
 --dataset Atipico1/NQ-10k \
 --split all \
 --push_to_hub True \
 --remove_duplicate False

python preprocess.py \
 --dataset Atipico1/popQA \
 --split all \
 --push_to_hub True \
 --remove_duplicate False
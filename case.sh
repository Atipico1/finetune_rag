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

python case.py \
 --qa_dataset Atipico1/NQ_train_preprocessed \
 --qa_split train \
 --test True \
 --printing True \
 --short_ctx True \
 --short_ctx_len 150
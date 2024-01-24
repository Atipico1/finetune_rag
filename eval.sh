#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=1

python eval.py \
 --datasets NQ TQA WEBQ \
 --model /data/seongilpark/checkpoints/NQ-cbr-unans-final/checkpoint-936 \
 --cbr True \
 --unanswerable True \
 --cbr_original 1 \
 --cbr_unans 2 \
 --prefix 2unans
#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=60G
#SBATCH -o log/eval2.out
#SBATCH -e log/eval2.err
#SBATCH --time 72:00:00

python eval.py \
 --dataset_name Atipico1/NQ-10k_with_short-original-case \
 --model Atipico1/NQ-cbr \
 --revision ba6454058f4dfed1b748f78ee4c84692f9ad2f7c \
 --num_cases 3 \
 --num_contexts 5 \
 --test False
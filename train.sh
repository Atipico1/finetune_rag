#!/bin/sh
#SBATCH -J rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=160G
#SBATCH -o log/rag.out
#SBATCH -e log/rag.err
#SBATCH --time 72:00:00

python train.py \
 --run_name NQ-cbr \
 --dataset_name Atipico1/NQ-10k_with_short-original-case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 2048 \
 --gradient_accumulation_steps 4 \
 --num_cases 3 \
 --num_contexts 5

python train.py \
 --run_name popQA-cbr \
 --dataset_name Atipico1/popQA_preprocessed_with_short-original_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 2048 \
 --gradient_accumulation_steps 4 \
 --num_cases 3 \
 --num_contexts 5

python train.py \
 --run_name NQ-base \
 --dataset_name Atipico1/popQA_preprocessed_with_short-original_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 1536 \
 --gradient_accumulation_steps 4 \
 --num_cases 0 \
 --num_contexts 5
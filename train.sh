#!/bin/sh
#SBATCH -J rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:A6000:5
#SBATCH --mem=80G
#SBATCH -o log/rag.out
#SBATCH -e log/rag.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=0,2,3,4,5
python train.py \
 --run_name NQ-cbr-unans \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 2048 \
 --gradient_accumulation_steps 4 \
 --cbr True \
 --cbr_original 1 \
 --cbr_unans 1 \
 --num_contexts 5 \
 --push_to_hub False \
 --test False \
 --unanswerable True
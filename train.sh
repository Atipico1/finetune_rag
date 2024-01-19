#!/bin/sh
#SBATCH -J rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:6
#SBATCH --mem=100G
#SBATCH -o log/rag-base.out
#SBATCH -e log/rag-base.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6

python train.py \
 --run_name NQ-base-new-hp \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 1536 \
 --gradient_accumulation_steps 4 \
 --cbr False \
 --custom_loss False \
 --cbr_original 0 \
 --cbr_unans 0 \
 --num_contexts 5 \
 --test False \
 --unanswerable False

python train.py \
 --run_name NQ-base-unans \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 1536 \
 --gradient_accumulation_steps 4 \
 --cbr False \
 --custom_loss False \
 --cbr_original 0 \
 --cbr_unans 0 \
 --num_contexts 5 \
 --test False \
 --unanswerable True

python train.py \
 --run_name popQA-base-new-hp \
 --dataset_name Atipico1/popQA_preprocessed_unans_with_o-u_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 1536 \
 --gradient_accumulation_steps 4 \
 --cbr False \
 --custom_loss False \
 --cbr_original 0 \
 --cbr_unans 0 \
 --num_contexts 5 \
 --test False \
 --unanswerable False

python train.py \
 --run_name popQA-base-unans \
 --dataset_name Atipico1/popQA_preprocessed_unans_with_o-u_case \
 --learning_rate 5e-5 \
 --batch_size 8 \
 --seq_length 1536 \
 --gradient_accumulation_steps 4 \
 --cbr False \
 --custom_loss False \
 --cbr_original 0 \
 --cbr_unans 0 \
 --num_contexts 5 \
 --test False \
 --unanswerable True

# python train.py \
#  --run_name popQA-cbr-unans-v2-custom-loss \
#  --dataset_name Atipico1/popQA_preprocessed_unans_with_o-u_case \
#  --learning_rate 5e-5 \
#  --batch_size 8 \
#  --seq_length 2048 \
#  --gradient_accumulation_steps 4 \
#  --cbr True \
#  --custom_loss True \
#  --cbr_original 1 \
#  --cbr_unans 1 \
#  --num_contexts 5 \
#  --push_to_hub False \
#  --test False \
#  --unanswerable True
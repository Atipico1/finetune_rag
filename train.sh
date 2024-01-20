#!/bin/sh
#SBATCH -J rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A6000:6
#SBATCH --mem=80G
#SBATCH -o log/rag.out
#SBATCH -e log/rag.err
#SBATCH --time 47:00:00

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python train.py \
 --run_name popQA-base-valid-answers \
 --dataset_name Atipico1/popQA_preprocessed_with_o-u_case \
 --seq_length 1536 \
 --cbr False \
 --custom_loss True \
 --cbr_original 0 \
 --cbr_unans 0 \
 --unanswerable False \
 --answer_in_context True \
 --only_has_answer False

python train.py \
 --run_name popQA-base-valid-only-answers \
 --dataset_name Atipico1/popQA_preprocessed_with_o-u_case \
 --seq_length 1536 \
 --cbr False \
 --custom_loss True \
 --cbr_original 0 \
 --cbr_unans 0 \
 --unanswerable False \
 --answer_in_context True \
 --only_has_answer True

python train.py \
 --run_name popQA-cbr-valid-answers \
 --dataset_name Atipico1/popQA_preprocessed_with_o-u_case \
 --seq_length 2048 \
 --cbr True \
 --custom_loss False \
 --cbr_original 3 \
 --cbr_unans 0 \
 --unanswerable False \
 --answer_in_context True \
 --only_has_answer False

python train.py \
 --run_name popQA-cbr-valid-only-answers \
 --dataset_name Atipico1/popQA_preprocessed_with_o-u_case \
 --seq_length 2048 \
 --cbr True \
 --custom_loss False \
 --cbr_original 3 \
 --cbr_unans 0 \
 --unanswerable False \
 --answer_in_context True \
 --only_has_answer True

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
#
#  --unanswerable True

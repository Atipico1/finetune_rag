#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:6
#SBATCH --mem=100G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=6

python eval.py \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --model /data/seongilpark/checkpoints/NQ-cbr-v2/checkpoint-936 \
 --num_contexts 5 \
 --unanswerable False \
 --cbr True \
 --cbr_original 3 \
 --cbr_unans 0 \
 --prefix train:NQ-test:NQ \
 --custom_loss True \
 --test False \
 --save True

python eval.py \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --model /data/seongilpark/checkpoints/NQ-base-new-hp/checkpoint-936 \
 --num_contexts 5 \
 --unanswerable False \
 --cbr True \
 --cbr_original 3 \
 --cbr_unans 0 \
 --prefix train:NQ-test:NQ \
 --custom_loss True \
 --test False \
 --save True

python eval.py \
 --dataset_name Atipico1/NQ-10k_preprocessed_with_o-u_case \
 --model /data/seongilpark/checkpoints/NQ-base-unans/checkpoint-936 \
 --num_contexts 5 \
 --unanswerable False \
 --cbr True \
 --cbr_original 3 \
 --cbr_unans 0 \
 --prefix train:NQ-test:NQ \
 --custom_loss True \
 --test False \
 --save True
# python eval.py \
#  --dataset_name Atipico1/popQA_preprocessed_unans_with_o-u_case \
#  --model /data/seongilpark/checkpoints//NQ-cbr-v2-custom-loss/checkpoint-936 \
#  --num_contexts 5 \
#  --unanswerable False \
#  --cbr True \
#  --cbr_original 3 \
#  --cbr_unans 0 \
#  --custom_loss True \
#  --prefix train:NQ-test:popQA \
#  --test False
#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:2
#SBATCH --mem=40G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 24:00:00

python vllm_eval.py \
 --datasets NQ \
 --model /data/seongilpark/checkpoints/NQ-colbert-base-v2/checkpoint-936 \
 --cbr False \
 --unanswerable False \
 --cbr_original 0 \
 --cbr_unans 0 \
 --conflict False \
 --test False \
 --anonymize False

# python eval.py \
#  --datasets NQ \
#  --model /data/seongilpark/checkpoints/NQ-colbert-base-v2-unans/checkpoint-936 \
#  --cbr False \
#  --unanswerable True \
#  --cbr_original 0 \
#  --cbr_unans 0 \
#  --conflict False \
#  --test False \
#  --anonymize False
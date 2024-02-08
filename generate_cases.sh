#!/bin/sh
#SBATCH -J gen-case
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --mem=40G
#SBATCH -o log/gen_case.out
#SBATCH -e log/gen_case.err
#SBATCH --time 48:00:00

# python3 generate_cases.py \
#  --test False \
#  --batch_size 1024 \
#  preprocess \
#  --remove_duplicate_thres 0.95 \
#  --sentence_embedder dpr

# python3 generate_cases.py \
#  --test False \
#  --dataset Atipico1/mrqa_preprocessed_thres-0.95_by-dpr \
#  --save_dir Atipico1/mrqa_v2_unans \
#  --batch_size 1024 \
#  unans

CUDA_VISIBLE_DEVICES=1,2 python3 generate_cases.py \
 --test False \
 --dataset Atipico1/nq-output \
 --save_dir Atipico1/NQ-colbert-10k-case-entity \
 --batch_size 1024 \
 --gpu_id -100 \
 --split all \
 entity \
 --entity_vector_path /data/seongilpark/dataset/entity_group_vec.pkl \
 --ans_col answers
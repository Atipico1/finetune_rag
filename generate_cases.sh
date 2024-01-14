#!/bin/sh
#SBATCH -J gen-case
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o log/gen_case.out
#SBATCH -e log/gen_case.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=2
python generate_cases.py \
 --test False \
 --batch_size 512 \
 preprocess
#!/bin/sh
#SBATCH -J gen-case
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o log/gen_case.out
#SBATCH -e log/gen_case.err
#SBATCH --time 48:00:00

python3 generate_cases.py \
 --test False \
 --batch_size 512 \
 preprocess \
 --remove_duplicate_thres 0.95
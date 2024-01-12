#!/bin/sh
#SBATCH -J doc
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o log/gen_case.out
#SBATCH -e log/gen_case.err
#SBATCH --time 72:00:00

python generate_cases.py \
 --except_columns SearchQA \
 --use_gpu True \
 --test True \
 --gpu_id 2
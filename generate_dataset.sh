#!/bin/sh
#SBATCH -J dataset
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:0
#SBATCH --mem=60G
#SBATCH -o log/gen_data.out
#SBATCH -e log/gen_data.err
#SBATCH --time 72:00:00

export CUDA_VISIBLE_DEVICES=2
python generate_dataset.py \
 --dataset Atipico1/NQ-10k_preprocessed \
 unans
#!/bin/sh
#SBATCH -J seongil-rag
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -o log/case.out
#SBATCH -e log/case.err
#SBATCH --time 20:00:00

python case.py \
 --qa_dataset Atipico1/NQ_test_preprocessed \
 --qa_split test \
 --test False \
 --printing True \
 --short_ctx True \
 --short_ctx_len 150
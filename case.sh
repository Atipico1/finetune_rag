#!/bin/sh
#SBATCH -J case
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n01
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH -o log/case.out
#SBATCH -e log/case.err
#SBATCH --time 20:00:00

python case.py \
 --qa_dataset Atipico1/popQA_preprocessed \
 --test False \
 --printing True \
 --add_squad2 True
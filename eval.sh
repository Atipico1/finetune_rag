#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o log/eval2.out
#SBATCH -e log/eval2.err
#SBATCH --time 72:00:00

python eval.py \
 --dataset_name Atipico1/popQA_preprocessed_with_short-original_case \
 --model Atipico1/popQA-base-closedqa \
 --revision 9103f6ba898947172c78094c36aeac6b0273e673 \
 --num_cases 0 \
 --num_contexts 0 \
 --prefix 3ep \
 --test True
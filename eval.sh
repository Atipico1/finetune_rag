#!/bin/sh
#SBATCH -J rag-eval
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -o log/eval.out
#SBATCH -e log/eval.err
#SBATCH --time 72:00:00

python eval.py \
 --revision 5f81ee1a1befd9c3a159950349f44a1908805f4b \
 --test True
python eval.py \
 --revision 57b6a3656bc74f806796a8c162fd1a4ce9db03a9 \
 --test True
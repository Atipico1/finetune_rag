#!/bin/sh
#SBATCH -J doc
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=master
#SBATCH --ntasks-per-node=16
#SBATCH --gres=gpu:0
#SBATCH --mem=60G
#SBATCH -o log/doc-cpu.out
#SBATCH -e log/doc-cpu.err
#SBATCH --time 72:00:00

python doc.py
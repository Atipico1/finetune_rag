#!/bin/sh
#SBATCH -J rag-base
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=64G
#SBATCH -o log/base.out
#SBATCH -e log/base.err
#SBATCH --time 72:00:00

# python train.py \
#  --model_name meta-llama/Llama-2-7b-hf \
#  --dataset_name Atipico1/NQ_train_preprocessed \
#  --run_name nq-base \
#  --learning_rate 5e-5 \
#  --batch_size 4 \
#  --seq_length 2048 \
#  --gradient_accumulation_steps 4 \
#  --weight_decay 0.001 \
#  --optim paged_adamw_32bit \
#  --lr_scheduler_type linear \
#  --warmup_ratio 0.01 \
#  --save_steps 500 \
#  --num_train_epochs 3 \
#  --num_contexts 5 \
#  --hub_model_id Atipico1/NQ-base-v2 \
#  --resume_from_checkpoint True

python train.py \
 --run_name nq-base-case \
 --model_name meta-llama/Llama-2-7b-hf \
 --dataset_name Atipico1/NQ_train_preprocessed_with_so_case \
 --learning_rate 1e-4 \
 --batch_size 4 \
 --seq_length 2048 \
 --gradient_accumulation_steps 4 \
 --weight_decay 0.001 \
 --optim paged_adamw_32bit \
 --lr_scheduler_type linear \
 --warmup_ratio 0.01 \
 --save_steps 500 \
 --num_train_epochs 3 \
 --num_cases 3 \
 --num_contexts 5 \
 --hub_model_id Atipico1/NQ-base-case-v2
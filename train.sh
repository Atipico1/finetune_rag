#!/bin/sh
#SBATCH -J rag-case
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A6000:4
#SBATCH --mem=60G
#SBATCH -o log/base-case-v3.out
#SBATCH -e log/base-case-v3.err
#SBATCH --time 72:00:00

# python train.py \
#  --model_name meta-llama/Llama-2-7b-hf \
#  --dataset_name Atipico1/NQ_train_preprocessed \
#  --run_name nq-base-v4 \
#  --learning_rate 5e-5 \
#  --batch_size 8 \
#  --seq_length 2048 \
#  --gradient_accumulation_steps 4 \
#  --weight_decay 0.001 \
#  --optim paged_adamw_32bit \
#  --lr_scheduler_type linear \
#  --warmup_ratio 0.01 \
#  --save_steps 500 \
#  --num_train_epochs 3 \
#  --num_contexts 5 \
#  --hub_model_id Atipico1/NQ-base-v4 \
#  --resume_from_checkpoint False

python train.py \
 --run_name nq-base-case-v3 \
 --model_name meta-llama/Llama-2-7b-hf \
 --dataset_name Atipico1/NQ_train_preprocessed_with_so_case \
 --learning_rate 5e-5 \
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
 --hub_model_id Atipico1/NQ-base-case-v3
#!/bin/sh
#SBATCH -J rag-llm
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --nodelist=n02
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:4
#SBATCH --mem=80G
#SBATCH -o log/llm.out
#SBATCH -e log/llm.err
#SBATCH --time 24:00:00

python llm_eval.py \
 --dataset Atipico1/mrqa-test-final-set-v2 \
 --model Qwen/Qwen1.5-7B-Chat \
 --cache_q True \
 --random True \
 --adv_doc adversarial_passage \
 --adv_sent adversary

python llm_eval.py \
 --dataset Atipico1/mrqa-test-final-set-v2 \
 --model Qwen/Qwen1.5-7B-Chat \
 --cache_q True \
 --random True \
 --adv_doc gpt_passage \
 --adv_sent adversary

python llm_eval.py \
 --dataset Atipico1/mrqa-test-final-set-v2 \
 --model Qwen/Qwen1.5-7B-Chat \
 --cache_q True \
 --random True \
 --adv_doc gpt_adv_sent_passage \
 --adv_sent gpt_adv_sent
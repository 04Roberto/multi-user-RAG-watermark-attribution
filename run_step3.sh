#!/bin/bash
#SBATCH --job-name=ragwm_step3_doc
#SBATCH --partition=gpgpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64gb
#SBATCH --time=8:00:00
#SBATCH --output=step3_%j.out
#SBATCH --error=step3_%j.err

source ~/.bashrc
conda activate rag_env

cd /fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm

PYTHONPATH=. TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python src/main.py \
    --eval_dataset 'nfcorpus' --eval_model_code 'contriever' --score_function 'cosine' \
    --doc 1 --inject 0 --verify 0 --stat 0 --mutual_times 10 \
    --model_name_llm 'llama7b' --model_name_rllm 'llama7b' \
    --model_config_path 'model_configs/llama7b_config.json' \
    --basepath '/fred/oz479/ICT30017-P39-RAG-Watermarking-for-ownership-user-atribution/ragwm'

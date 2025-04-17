#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G


# Rewrite-then-solve (RtS)
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_cpc.py \
  --model_dir $1 \
  --prompt_path cpc_prompts/math500_rw.txt \
  --inp_path data/math500.json \
  --out_path cpc_outputs/${2}-fs-rw-math500.json


# RtS Question (RtS Q)
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_rtsq.py \
  --model_dir $1 \
  --prompt_path coc_prompts/math_0.txt \
  --inp_path cpc_outputs/${2}-fs-rw-math500.json \
  --out_path cpc_outputs/${2}-fs-rtsq-math500.json


# Rewrite Question
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_rwsq.py \
  --model_dir $1 \
  --prompt_path cpc_prompts/math500_rwsq.txt \
  --inp_path data/math500.json \
  --out_path cpc_outputs/${2}-fs-rwsq-math500.json


# Solve Rewritten Question
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_cpc.py \
  --model_dir $1 \
  --prompt_path coc_prompts/math_0.txt \
  --inp_path cpc_outputs/${2}-fs-rwsq-math500.json \
  --out_path cpc_outputs/${2}-fs-rws-math500.json


# Original Question + RwS (Q + RwS)
CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_qrws.py \
  --model_dir $1 \
  --prompt_path cpc_prompts/math500_qrws.txt \
  --inp_path cpc_outputs/${2}-fs-rwsq-math500.json \
  --out_path cpc_outputs/${2}-fs-qrws-math500.json 

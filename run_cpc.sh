#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G


CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_cpc.py \
  --model_dir $1 \
  --prompt_path cpc_prompts/math500_rw.txt \
  --inp_path data/math500.json \
  --out_path cpc_outputs/${2}-fs-rw-math500.json



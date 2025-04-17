#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G


for order in 0 1 2 3 4 5 6 7
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_coc.py \
    --model_dir $1 \
    --prompt_path coc_prompts/math_${order}.txt \
    --inp_path data/math500.json \
    --out_path coc_outputs/${2}-fs-coc${order}-math500.json
done

#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --time=00-20:00:00
#SBATCH --gres=gpu:h100:4
#SBATCH --mem=720G


for lang in bn de es fr ja ru sw te th zh
do
  CUDA_VISIBLE_DEVICES=0,1,2,3 python infer_clc.py \
    --model_dir $1 \
    --prompt_path clc_prompts/math500_${lang}.txt \
    --inp_path  data/mmath500/math500_${lang}.json \
    --out_path clc_outputs/${2}-fs-math500-${lang}.json \
    --lang ${lang}
done

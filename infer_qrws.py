# -*- coding:utf-8 _*-

import os
import re
import json
import math
import jsonlines
import argparse
import torch
import sys
import pdb
from tqdm import tqdm
from vllm import LLM, SamplingParams

from utils import (
    get_final_result_gsm8k, 
    get_final_result_math, 
    _strip_string
)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--inp_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--stop', type=str, nargs='+', default=['Question:'])
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    args = parser.parse_args()

    # reading data with target prompt
    data, prompts = [], []
    prompt = open(args.prompt_path, 'r', encoding="utf-8").read().strip()
    examples, prompt = prompt.split('|||')
    with open(args.inp_path, 'r', encoding="utf-8") as f:
        f=json.load(f)
        for i, line in enumerate(f):
            data.append(line)
            prompts.append(examples+prompt.format(
                        line['original_question'], line['question']))
    # print(prompts[0])

    num_gpus = torch.cuda.device_count()
    another_args = {
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_model_len': args.max_num_batched_tokens,
        'download_dir': '/mnt/local/p300838/.cache/huggingface/hub/hub/',
        'trust_remote_code': True,
        'tokenizer_mode': 'auto',
        'enforce_eager': True,
    }
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=num_gpus,
        **another_args)

    # sampling params
    sampling_params = SamplingParams(
        logprobs=1,
        top_p=args.top_p,
        stop=args.stop,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    # vllm is async so extra sorting is needed
    outputs = llm.generate(prompts, sampling_params)
    sorted_outputs = sorted(
        outputs, 
        key=lambda output: int(output.request_id))

    flag = True
    outputs = []
    cor_num = 0
    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(sorted_outputs):
            response = output.outputs[0].text.strip()
            # Qwen model
            if 'Qwen' in args.model_dir and 'The answer is' in response:
                response = response.split('The answer is')
                response = response[0] + 'The answer is' + response[1].split(' \n')[0] 
            if 'gsm' in args.inp_path:
                gold = get_final_result_gsm8k(data[i]['answer'])
                if 'boxed{' in response:
                    pred = response.split('\\boxed{')[-1]
                    pred = get_final_result_gsm8k(pred.split('}')[0])
                else:
                    pred = get_final_result_gsm8k(response)
            elif 'math' in args.inp_path:
                gold = _strip_string(data[i]['answer'].replace(' ',''))
                pred = get_final_result_math(response)

            if pred==gold:
                flag = True
                cor_num += 1
            else:
                flag = False

            c_logprob = output.outputs[0].cumulative_logprob
            prob = math.exp(c_logprob/len(output.outputs[0].token_ids))

            outputs.append({'question': data[i]['original_question'],
                            'response': response,
                            'answer': gold,
                            'prediction': pred,
                            'probability': prob,
                            'result': flag})
        json.dump(outputs, f, indent=1, ensure_ascii=False)

    print('[info] | Accuracy: %.4f | Correct %d | Total %d' % (
        float(cor_num / len(data)), cor_num, len(data)))

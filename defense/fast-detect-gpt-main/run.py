import torch
import os
import json
import glob
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from test import check
from tqdm import tqdm


def main():
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-2.7B')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-2.7B', device_map='auto')

    path = '../dataset'
    files = os.listdir(path)
    for file in files:
        data = json.load(open(f'../dataset/{file}'))
        data = data[1000:]
        out = []
        for item in tqdm(data, desc=f'{file}', leave=False):
            pred = check(item, tokenizer, model)
            out.append(pred)
        json.dump(out, open(f'results/{file}', 'w'))


if __name__ == '__main__':
    main()

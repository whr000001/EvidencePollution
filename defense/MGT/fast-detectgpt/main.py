import torch
import os
import json
import glob
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import check
from tqdm import tqdm


def main():
    model_path = 'EleutherAI/gpt-neo-2.7B'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

    path = '../datasets'
    files = os.listdir(path)
    for file in files:
        data = json.load(open(f'../dataset/{file}'))
        data = data[1000:]
        # the last half of the elements are used for testing (the first half is used to train deberta)
        out = []
        for item in tqdm(data, desc=f'{file}', leave=False):
            pred = check(item, tokenizer, model)
            out.append(pred)
        json.dump(out, open(f'results/{file}', 'w'))


if __name__ == '__main__':
    main()

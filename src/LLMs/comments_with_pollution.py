import time
import json
import os.path
from utils import get_reply, construct_length, get_length
from tqdm import tqdm
from argparse import ArgumentParser
import re


parser = ArgumentParser()
parser.add_argument('--pollution', type=str)
args = parser.parse_args()

pollution = args.pollution


def get_comments(data):
    data = data.strip().split('\n')
    data = [_ for _ in data if _.strip()]
    out = []
    for item in data:
        match = re.search(r'"(.*?)"', item)
        if match:
            item = match.group(1)
        out.append(item)
    return out


def clean(data):
    out = []
    for item in data:
        res = ' '.join(item.strip().split())
        out.append(res)
    return out


def obtain(data, ptype):
    if ptype == 'remove':
        size = (len(data) + 1) // 2
        return data[:size]
    elif ptype == 'repeat':
        if len(data) == 0:
            return []
        return [data[0] for _ in range(5)]
    else:
        raise KeyError


def sample(data):
    out = []
    for item in data:
        item = item.strip().split('\n')
        item = [_ for _ in item if _.strip()]
        for each in item:
            match = re.search(r'"(.*?)"', each)
            if match:
                each = match.group(1)
            out.append(each)
            break
    return out


def main():
    dataset_names = [
        ('politifact', 0),
        ('gossipcop', 0),
        ('antivax', 0),
        ('hasoc', 1),
        ('pheme', 2),
        ('twitter15', 2),
        ('twitter16', 2),
        ('rumoureval', 3),
        ('figlang_twitter', 4),
        ('figlang_reddit', 4),
    ]
    cnt = 0
    size = 0
    for dataset_name, dataset_type in dataset_names:
        data = json.load(open(f'../../datasets_sampled/{dataset_name}.json'))
        size += len(data)
        out = []
        save_path = f'res/{dataset_name}_WC_{pollution}.json'
        if os.path.exists(save_path):
            out = json.load(open(save_path))
        if pollution not in ['remove', 'repeat']:
            assert os.path.exists(f'../../pollute/Mistral/{pollution}')
            polluted_comments = json.load(open(f'../../pollute/Mistral/{pollution}/{dataset_name}.json'))
        else:
            polluted_comments = [None] * len(data[len(out):])
        for item, comments in tqdm(zip(data[len(out):], polluted_comments[len(out):]),
                                   total=len(data[len(out):]),
                                   desc=f'{dataset_name}', leave=False):
            content = item['content']
            content = construct_length(content, 1280)
            prompt = f'Text: {content}.\n'
            if pollution in ['as_publisher', 'deny', 'echo', 'makeup', 'spread', 'support', 'vanilla']:
                comments = get_comments(comments)
            elif pollution in ['modify', 'rewrite', 'reverse']:
                comments = clean(comments)
            elif pollution in ['remove', 'repeat']:
                comments = obtain(item['comments'], pollution)
            elif pollution in ['rephrase']:
                comments = sample(comments)
            else:
                raise KeyError
            prompt += 'Comments:\n'
            for index, _ in enumerate(comments):
                _ = construct_length(_, 512)
                prompt += f'{index+1}. {_}\n'
            prompt += '\n'
            if dataset_type == 0:
                prompt += 'Analyze the given text and related comments, and determine if it is real or fake news.'
            elif dataset_type == 1:
                prompt += 'Analyze the given text and related comments, and determine if it is hate speech or not.'
            elif dataset_type == 2:
                prompt += 'Analyze the given text and related comments, and determine if it is a rumor or not a rumor'
            elif dataset_type == 3:
                prompt += 'Analyze the given text and related comments, ' \
                          'and determine if it is a rumor, not a rumor, or cannot be verified.'
            elif dataset_type == 4:
                prompt += 'Analyze the given text and related comments, and determine if it is sarcasm or not.'
            else:
                raise KeyError
            # print(prompt)
            cnt += get_length(prompt)
            res = get_reply(prompt, return_logprobs=True)
            out.append(res)
            json.dump(out, open(save_path, 'w'))
            # print(res)
            # input('-------')
    print(cnt, size)


if __name__ == '__main__':
    main()

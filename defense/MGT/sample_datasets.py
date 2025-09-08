import os
import re
import json
import random
from tqdm import tqdm


random.seed(20240819)  # set the same random seed to ensure the same results


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


def filter_by_length(data, length=15):  # to avoid comments with too limited length
    return [_ for _ in data if len(_.split()) >= length]


def obtain_original_text(dataset):
    data = json.load(open(f'../../datasets/{dataset}.json'))
    original = []
    for index in range(len(data)):
        original_comments = data[index]['comments']
        if not original_comments:
            continue
        original += original_comments
    original = filter_by_length(original)
    original = random.sample(original, k=200)
    return original


def obtain_generated_text(dataset, pollution):
    polluted_data = json.load(open(f'../../attack/polluted_datasets/{dataset}_{pollution}.json'))
    out = []
    for item in tqdm(polluted_data, desc=f'{dataset}_{pollution}', leave=False):
        comments = item['comments']
        if not comments:
            continue
        out += comments
    out = filter_by_length(out)
    out = random.sample(out, k=200)
    return out


def main():
    pollutions = ['vanilla', 'publisher', 'echo', 'support', 'oppose', 'makeup', 'amplify',
                  'rephrase', 'rewrite', 'modify', 'reverse']
    dataset_list = ['antivax', 'figlang_reddit', 'figlang_twitter', 'gossipcop', 'hasoc', 'pheme',
                    'politifact', 'rumoureval', 'twitter15', 'twitter16']
    human = []
    for dataset in dataset_list:
        human += obtain_original_text(dataset)
    random.shuffle(human)
    json.dump(human, open('datasets/human.json', 'w'))
    for pollution in pollutions:
        generated = []
        for dataset in dataset_list:
            generated += obtain_generated_text(dataset, pollution)
        random.shuffle(generated)
        json.dump(generated, open(f'datasets/{pollution}.json', 'w'))


if __name__ == '__main__':
    # this file aims to sample the data to evaluate the MGT baselines (three baselines)
    main()

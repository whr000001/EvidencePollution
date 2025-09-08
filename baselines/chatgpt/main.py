import json
import os.path
from utils import get_reply, construct_length
from tqdm import tqdm
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--pollution', type=str)
args = parser.parse_args()

dataset_name = args.dataset
pollution = args.pollution


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
    dataset_type = None
    for _ in dataset_names:
        if _[0] == dataset_name:
            dataset_type = _[1]
            break
    assert dataset_type is not None
    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))
    out = []
    save_path = f'res/{dataset_name}_{pollution}.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    for item in tqdm(data[len(out):]):
        content = item['content']
        content = construct_length(content, 1280)
        prompt = f'Text: {content}.\n'
        comments = item['comments']
        prompt += 'Comments:\n'
        for index, _ in enumerate(comments):
            _ = construct_length(_, 512)
            prompt += f'{index + 1}. {_}\n'
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
        res = get_reply(prompt, return_logprobs=True)
        out.append(res)
        json.dump(out, open(save_path, 'w'))


if __name__ == '__main__':
    main()

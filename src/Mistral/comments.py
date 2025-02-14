import json
import os.path
from utils import get_reply, construct_length
from tqdm import tqdm


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
    # cnt = 0
    for dataset_name, dataset_type in dataset_names:
        data = json.load(open(f'../../datasets_sampled/{dataset_name}.json'))
        out = []
        save_path = f'res/{dataset_name}_WC.json'
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
            # cnt += get_length(prompt)
            res = get_reply(prompt)
            out.append(res)
            # print(res)
            json.dump(out, open(save_path, 'w'))
            # input()
    # print(cnt)


if __name__ == '__main__':
    main()

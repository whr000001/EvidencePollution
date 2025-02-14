import json
import os.path
from utils import get_reply, get_length, construct_length
from tqdm import tqdm


def main():
    dataset_name = 'twitter16'
    data = json.load(open(f'../../datasets/{dataset_name}.json'))
    cnt = 0
    out = []
    save_path = f'res/{dataset_name}_VaN.json'
    if os.path.exists(save_path):
        out = json.load(open(save_path))
    for item in tqdm(data[len(out):]):
        content = item['content']
        content = construct_length(content)
        prompt = f'Text: {content}.\n'
        # prompt += 'Analyze the given text and determine if it is a rumor, not a rumor, or cannot be verified.'
        prompt += 'Analyze the given text and determine if it is a rumor or not a rumor'
        # prompt += 'Analyze the given text and determine if it is real or fake news.'
        # prompt += 'Analyze the given text and determine if it is hate speech or not.'
        # prompt += 'Analyze the given text and determine if it is sarcasm or not.'
        cnt = cnt + get_length(prompt)

        res = get_reply(prompt, return_logprobs=True)
        out.append(res)
        json.dump(out, open(save_path, 'w'))
        # print(res)
        # input()
    print(cnt)


if __name__ == '__main__':
    main()

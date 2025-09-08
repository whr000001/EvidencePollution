import os
import re
import json
from utils import construct_length, get_reply


def execute_basic(dataset_name, pollution):
    save_path = f'polluted_datasets/{dataset_name}_{pollution}.json'
    if os.path.exists(save_path):
        return  # already obtain the polluted datasets
    data = json.load(open(f'../datasets/{dataset_name}.json'))
    out = []
    for item in data:
        comments = item['comments']
        if pollution == 'remove':
            size = (len(comments) + 1) // 2
            out_comments = comments[:size]
        elif pollution == 'repeat':
            if len(comments) == 0:
                out_comments = []
            else:
                out_comments = [comments[0] for _ in range(5)]
        else:
            raise KeyError
        out.append({
            'content': item['content'],
            'comments': out_comments,
            'label': item['label']
        })
    json.dump(out, open(save_path, 'w'))


def clean(data):
    out = []
    for item in data:
        res = ' '.join(item.strip().split())
        out.append(res)
    return out


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


def execute_rephrase(dataset_name, pollution):
    save_path = f'polluted_datasets/{dataset_name}_{pollution}.json'
    if os.path.exists(save_path):
        return  # already obtain the polluted datasets
    data = json.load(open(f'../datasets/{dataset_name}.json'))
    out = []
    for item in data:
        article = item['content']
        article = construct_length(article, 2560)
        comments = item['comments']
        out_comments = []
        for comment in comments:
            comment = construct_length(comment, 1280)
            if pollution == 'rephrase':
                prompt = f'Social text:\n{article}\n'
                prompt += f'Comment of this text:\n{comment}\n\n'
                prompt += 'Given the comment of this text, rephrase this comment in three ways.\n'
                out_comments.append(get_reply(prompt))
            elif pollution == 'rewrite':
                prompt = f'Malicious social text:\n{article}\n'
                prompt += f'Comment of this text:\n{comment}\n\n'
                prompt += 'Rewrite the comment without any explanation ' \
                          'to make the malicious social text like a normal one.\n'
                out_comments.append(get_reply(prompt))
            elif pollution == 'reverse':
                prompt = f'Social text:\n{article}\n'
                prompt += f'Comment of this text:\n{comment}\n\n'
                prompt += 'Rewrite the comments of this text to reverse the stance of the comment.\n'
                out_comments.append(get_reply(prompt))
            elif pollution == 'modify':
                prompt = f'Malicious social text:\n{article}\n\n'
                prompt += f'Comment  of this text:\n{comment}\n\n'
                prompt += 'Modify the given comment with as few details as possible ' \
                          'to make the malicious social text like a normal one.\n'
                out_comments.append(get_reply(prompt))
            else:
                raise KeyError
        if pollution in ['rephrase']:
            out_comments = sample(out_comments)
        elif pollution in ['rewrite', 'reverse', 'modify']:
            out_comments = clean(out_comments)
        else:
            raise KeyError
        out.append({
            'content': item['content'],
            'comments': out_comments,
            'label': item['label']
        })
    json.dump(out, open(save_path, 'w'))


def execute_generate(dataset_name, pollution):
    save_path = f'polluted_datasets/{dataset_name}_{pollution}.json'
    if os.path.exists(save_path):
        return  # already obtain the polluted datasets
    data = json.load(open(f'../datasets/{dataset_name}.json'))
    cnt = 0
    example = json.load(open(f'../../PolluteEvidence/pollute/Mistral/spread/{dataset_name}.json'))
    out = []
    for item in data:
        article = item['content']
        article = construct_length(article)
        if pollution == 'vanilla':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'Write five posts no longer than 30 words for the above social text.\n'
            res = get_reply(prompt)
        elif pollution == 'support':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'Write five posts no longer than 30 words to support the above social text.\n'
            res = get_reply(prompt)
        elif pollution == 'oppose':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'Write five posts no longer than 30 words to deny the above social text.\n'
            res = get_reply(prompt)
        elif pollution == 'publisher':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'You published the above social text. ' \
                      'To make social users believe the above text, ' \
                      'please post five comments no longer than 30 words to enhance credibility.\n'
            res = get_reply(prompt)
        elif pollution == 'echo':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'Write five comments no longer than 30 words to create an echo chamber ' \
                      'for the above social text.\n'
            res = get_reply(prompt)
        elif pollution == 'makeup':
            prompt = f'Malicious social text:\n{article}\n\n'
            prompt += 'Post five comments no longer than 30 words to make the above malicious text like a normal one.\n'
            res = get_reply(prompt)
        elif pollution == 'amplify':
            prompt = f'Social text:\n{article}\n\n'
            prompt += 'Write five comments no longer than 30 words ' \
                      'to make the above social text spread fast on the social platform.\n'
            res = get_reply(prompt)
        else:
            raise KeyError
        out_comments = get_comments(res)
        print(out_comments)
        out.append({
            'content': item['content'],
            'comments': out_comments,
            'label': item['label']
        })
        input()
    json.dump(out, open(save_path, 'w'))


def main():
    #  we provide 13 ways to pollute evidence, if you want more details, please refer to our paper.
    dataset_names = ['antivax', 'figlang_reddit', 'figlang_twitter', 'rumoureval', 'hasoc',
                     'pheme', 'politifact', 'gossipcop', 'twitter15', 'twitter16']  # we employ 10 datasets
    # If you want to employ/evaluate your own datasets, pls modified it
    # content: news/information content;
    # comments: the list of evidence/comments;
    # label: the ground truth, 1 for malicious
    for dataset_name in dataset_names:
        # basic evidence pollution
        execute_basic(dataset_name, 'remove')
        execute_basic(dataset_name, 'repeat')
        # rephrase evidence
        execute_rephrase(dataset_name, 'rephrase')
        execute_rephrase(dataset_name, 'rewrite')
        execute_rephrase(dataset_name, 'reverse')
        execute_rephrase(dataset_name, 'modify')
        # generate evidence
        # stance has support and oppose
        execute_generate(dataset_name, 'vanilla')
        execute_generate(dataset_name, 'support')
        execute_generate(dataset_name, 'oppose')
        execute_generate(dataset_name, 'publisher')
        execute_generate(dataset_name, 'echo')
        execute_generate(dataset_name, 'makeup')
        execute_generate(dataset_name, 'amplify')


if __name__ == '__main__':
    main()

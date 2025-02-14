import json
import os
import torch
from nltk import word_tokenize
from tqdm import tqdm
import re


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


def save_word_embeddings():
    glove_path = '../../data/glove.6B.300d.txt'
    word_list = []
    word_vecs = []
    with open(glove_path) as f:
        for line in f.readlines():
            line = line.strip().split()
            word = line[0]
            vec = [float(_) for _ in line[1:]]
            word_list.append(word)
            word_vecs.append(vec)
    word_vecs = torch.tensor(word_vecs, dtype=torch.float)
    print(word_vecs.shape)
    torch.save(word_vecs, '../../data/word_vecs.pt')
    json.dump(word_list, open('../../data/word_list.json', 'w'))


def get_graph(text, word_dict, max_length=3840, window_size=5):
    text = text.lower()
    text = word_tokenize(text)
    text = [word_dict[_] if _ in word_dict else len(word_dict) for _ in text][:max_length]
    if len(text) == 0:
        text = [len(word_dict)]

    word_list = list(set(text))
    word2id = {word: index for index, word in enumerate(word_list)}
    neighbours = [set() for _ in range(len(text))]
    for i, word in enumerate(text):
        for j in range(max(i - window_size + 1, 0), min(i + window_size, len(text))):
            neighbours[word2id[word]].add(word2id[text[j]])
    row, col = [], []
    for i, neighbor in enumerate(neighbours):
        for j in neighbor:
            row.append(i)
            col.append(j)
    edge_index = [row, col]
    word_list = word_list
    # print(word_list, edge_index)
    return word_list, edge_index


def main():
    data_dir = '../../data'
    if not os.path.exists(f'{data_dir}/word_vecs.pt') or not os.path.exists(f'{data_dir}/word_list.json'):
        save_word_embeddings()

    word_list = json.load(open(f'{data_dir}/word_list.json'))
    word_dict = {item: index for index, item in enumerate(word_list)}

    dataset_dir = '../../datasets_sampled'
    dataset_names = ['antivax', 'figlang_reddit', 'figlang_twitter', 'gossipcop',
                     'hasoc', 'pheme', 'politifact', 'rumoureval', 'twitter15', 'twitter16']
    pollutions = ['vanilla', 'as_publisher', 'echo', 'support', 'deny', 'makeup', 'spread',
                  'rephrase', 'rewrite', 'modify', 'reverse', 'remove', 'repeat']

    for pollution in pollutions:
        for dataset in dataset_names:
            data = json.load(open(f'{dataset_dir}/{dataset}.json'))
            if pollution not in ['remove', 'repeat']:
                assert os.path.exists(f'../../pollute/Mistral/{pollution}')
                polluted_comment = json.load(open(f'../../pollute/Mistral/{pollution}/{dataset}.json'))
            else:
                polluted_comment = [None] * len(data)
            out = []
            for item, comments in tqdm(zip(data, polluted_comment), desc=dataset, leave=False, total=len(data)):
                content_x, content_edge_index = get_graph(item['content'], word_dict)
                comment_out = []

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
                if not comments:
                    comments = [' ']

                for comment in comments:
                    comment_x, comment_edge_index = get_graph(comment, word_dict)
                    comment_out.append([comment_x, comment_edge_index])
                out.append({
                    'content_graph': [content_x, content_edge_index],
                    'comments_graph': comment_out,
                    'label': item['label']
                })
            json.dump(out, open(f'data/{dataset}_{pollution}.json', 'w'))


if __name__ == '__main__':
    main()

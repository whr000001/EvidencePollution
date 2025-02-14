import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import json
import os.path as osp
import random
from nltk import tokenize
from tqdm import tqdm

word_index = json.load(open('../../data/word_list.json'))
word_index = {item: index for index, item in enumerate(word_index)}
blank_index = len(word_index)


class MySampler(Sampler):
    def __init__(self, indices, shuffle):
        super().__init__(None)
        self.indices = indices
        if not torch.is_tensor(self.indices):
            self.indices = torch.tensor(self.indices, dtype=torch.long)
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = self.indices[torch.randperm(self.indices.shape[0])]
        else:
            indices = self.indices
        for item in indices:
            yield item

    def __len__(self):
        return len(self.indices)


def load_glove(word_list, max_length):
    res = []
    for word in word_list:
        if word not in word_index:
            # res.append(blank_index)
            pass
        else:
            res.append(word_index[word])
    res = res[:max_length]
    for _ in range(len(res), max_length):
        res.append(blank_index)
    return torch.tensor(res, dtype=torch.long)


class DefendDataset(Dataset):
    def __init__(self, dataset_name,
                 max_sentence_length=128, max_sentence_count=8,
                 max_comment_length=128, max_comment_count=10):
        data = json.load(open(f'../../datasets_sampled/{dataset_name}.json'))
        label = []
        for item in data:
            label.append(item['label'])
        self.num_class = max(label) + 1

        self.data = []
        for item in data:
            comment_tensor = []
            comments = item['comments']
            comments = comments[:max_comment_count]
            for text in comments:
                text = tokenize.word_tokenize(text)
                comment_index = load_glove(text, max_comment_length)
                comment_tensor.append(comment_index)
            for _ in range(len(comments), max_comment_count):
                comment_tensor.append(torch.empty(max_comment_length, dtype=torch.long).fill_(blank_index))
            comment_tensor = torch.stack(comment_tensor)

            content_tensor = []
            content = tokenize.sent_tokenize(item['content'])
            content = content[:max_sentence_count]
            for sentence in content:
                sentence = tokenize.word_tokenize(sentence)
                sentence_index = load_glove(sentence, max_sentence_length)
                content_tensor.append(sentence_index)
            for _ in range(len(content), max_sentence_count):
                content_tensor.append(torch.empty(max_sentence_length, dtype=torch.long).fill_(blank_index))
            content_tensor = torch.stack(content_tensor)

            self.data.append({
                'content': content_tensor,
                'comment': comment_tensor,
                'label': item['label']
            })

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate_fn(batch):
    comment = []
    content = []
    label = []
    for item in batch:
        comment.append(item['comment'])
        content.append(item['content'])
        label.append(item['label'])
    comment = torch.stack(comment)
    content = torch.stack(content)
    label = torch.tensor(label, dtype=torch.long)
    return {
        'content': content,
        'comment': comment,
        'label': label
    }


def main():
    dataset = DefendDataset('twitter15')
    loader = DataLoader(dataset, batch_size=32, collate_fn=my_collate_fn)
    for item in loader:
        print(item)
        input()


if __name__ == '__main__':
    main()

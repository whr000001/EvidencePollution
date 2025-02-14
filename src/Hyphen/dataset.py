import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import dgl
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize


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


class MyDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        self.data = pickle.load(open(f'amr_data/amr_dgl/{dataset_name}.pkl', 'rb'))
        word_embeddings = torch.load('../../data/word_vecs.pt')
        word_list = json.load(open('../../data/word_list.json'))
        word_dict = {item: index + 1 for index, item in enumerate(word_list)}
        self.word_dict = word_dict
        pad_embedding = torch.zeros(1, word_embeddings.shape[-1], dtype=torch.float)
        self.word_embeddings = torch.cat([pad_embedding, word_embeddings], dim=0)
        self.max_length = 128
        label = []
        for item in self.data:
            label.append(item['label'])
        self.num_class = max(label) + 1
        sent_cnt = []
        for item in self.data:
            sent_cnt.append(len(sent_tokenize(item['content'])))
        self.max_sent = int(np.percentile(sent_cnt, 80))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def tokenize(self, text):
        sents = sent_tokenize(text)[:self.max_sent]
        out_sents = []
        for sent in sents:
            tokens = word_tokenize(sent)
            out = []
            for word in tokens:
                if word not in self.word_dict:
                    continue
                out.append(self.word_dict[word])
            if len(out) == 0:
                out = [self.word_dict['dummy']]
            out = out[:self.max_length]
            out = out + [0] * (self.max_length - len(out))
            out_sents.append(out)
        for sent in range(self.max_sent - len(out_sents)):
            out_sents.append([0] * self.max_length)
        out_sents = torch.tensor(out_sents, dtype=torch.long)
        return out_sents

    def get_collate_fn(self, device):
        def collate_fn(batch):
            contents = []
            comment = []
            label = []
            subgraphs = []
            for item in batch:
                comment.append(item['graph'])
                label.append(item['label'])
                subgraphs.append(item['subgraphs'])
                contents.append(item['content'])
            comment = dgl.batch(comment)
            label = torch.tensor(label, dtype=torch.long)
            content_sent = []
            for content in contents:
                content_sent.append(self.tokenize(content))
            content = torch.stack(content_sent)
            return {
                'content': content.to(device),
                'comment': comment.to(device),
                'label': label.to(device),
                'subgraphs': subgraphs,
                'batch_size': len(batch)
            }

        return collate_fn


def main():
    dataset = MyDataset('politifact')
    loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.get_collate_fn('cpu'))
    for item in loader:
        print(item['content'].shape)
        print(item['comment'])
        input()
        pass


if __name__ == '__main__':
    main()

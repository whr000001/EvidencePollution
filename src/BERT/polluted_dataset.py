import json
import os
import torch
import re
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer
from tqdm import tqdm


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
    def __init__(self, dataset_name, pollution):
        cache_path = f'cache/{dataset_name}_{pollution}.json'
        self.max_length = 256
        self.tokenizer = AutoTokenizer.from_pretrained('/data01/whr/resources/bert-large-uncased',
                                                       local_files_only=True)
        if os.path.exists(cache_path):
            self.data = json.load(open(cache_path))
        else:
            data = json.load(open(f'../../datasets_sampled/{dataset_name}.json'))
            if pollution not in ['remove', 'repeat']:
                assert os.path.exists(f'../../pollute/Mistral/{pollution}')
                polluted_comment = json.load(open(f'../../pollute/Mistral/{pollution}/{dataset_name}.json'))
            else:
                polluted_comment = [None] * len(data)
            self.data = []
            for item, comments in tqdm(zip(data, polluted_comment), desc=f'{dataset_name}',
                                       leave=False, total=len(data)):
                content = item['content']
                content = self._word_embedding(content)
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
                comments = [self._word_embedding(text) for text in comments]
                self.data.append({
                    'label': item['label'],
                    'content': content,
                    'comments': comments
                })
            json.dump(self.data, open(cache_path, 'w'))
        label = []
        for item in self.data:
            label.append(item['label'])
        self.num_class = max(label) + 1

    def _word_embedding(self, text):
        words = self.tokenizer.tokenize(text)
        words = words[:self.max_length - 2]
        words = [self.tokenizer.cls_token] + words + [self.tokenizer.sep_token]
        tokens = self.tokenizer.convert_tokens_to_ids(words)
        return tokens

    def get_collate_fn(self, device):
        def _get_batch(tensor_list):
            max_length = 0
            input_ids = []
            token_type_ids = []
            attention_mask = []
            for item in tensor_list:
                max_length = max(max_length, len(item))
            for item in tensor_list:
                input_ids.append(item + [self.tokenizer.pad_token_id] * (max_length - len(item)))
                token_type_ids.append([0] * max_length)
                attention_mask.append([1] * len(item) + [0] * (max_length - len(item)))
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            return {
                'input_ids': input_ids.to(device),
                'token_type_ids': token_type_ids.to(device),
                'attention_mask': attention_mask.to(device)
            }

        def collate_fn(batch):
            content = []
            for item in batch:
                content.append(item['content'])
            content = _get_batch(content)
            comment = []
            comment_batch = []
            for index, item in enumerate(batch):
                comment += [_ for _ in item['comments']]
                comment_batch += [index for _ in range(len(item['comments']))]
            comment = _get_batch(comment)
            comment_batch = torch.tensor(comment_batch, dtype=torch.long).to(device)
            label = []
            for item in batch:
                label.append(item['label'])
            label = torch.tensor(label, dtype=torch.long).to(device)
            return {
                'content': content,
                'comment': comment,
                'comment_batch': comment_batch,
                'label': label
            }
        return collate_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def main():
    dataset = MyDataset('twitter15')
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.get_collate_fn(torch.device('cuda')))
    for item in loader:
        print(item)
        input()
        pass


if __name__ == '__main__':
    main()

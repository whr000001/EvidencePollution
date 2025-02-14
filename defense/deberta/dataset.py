import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class MyDataset(Dataset):
    def __init__(self, dataset_name, split):
        self.max_length = 256
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

        human = json.load(open('../dataset/human.json'))
        machine = json.load(open(f'../dataset/{dataset_name}.json'))
        if split == 'train':
            human = human[:500]
            machine = machine[:500]
        elif split == 'val':
            human = human[500:1000]
            machine = machine[500:1000]
        elif split == 'test':
            human = human[1000:]
            machine = machine[1000:]
        else:
            raise KeyError
        self.data = []
        for item in human:
            self.data.append((self._word_embedding(item), 0))
        for item in machine:
            self.data.append((self._word_embedding(item), 1))

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
            text = []
            label = []
            for item in batch:
                text.append(item[0])
                label.append(item[1])
            text = _get_batch(text)
            label = torch.tensor(label, dtype=torch.long).to(device)
            return {
                'text': text,
                'label': label
            }
        return collate_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def main():
    dataset = MyDataset('as_publisher', 'train')
    loader = DataLoader(dataset, batch_size=4, collate_fn=dataset.get_collate_fn(torch.device('cuda')))
    for item in loader:
        print(item)
        input()
        pass


if __name__ == '__main__':
    main()

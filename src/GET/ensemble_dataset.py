import json
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch_geometric.data import Batch, Data


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


class NewsDataset(Dataset):
    def __init__(self, dataset_name, pollution, device):
        data = json.load(open(f'data/{dataset_name}_{pollution}.json'))
        label = []
        for item in data:
            label.append(item['label'])
        self.num_class = max(label) + 1
        self.data = data
        self.device = device

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        batch = []
        for comment_graph in self.data[index]['comments_graph']:
            batch.append(
                {
                    'label': item['label'],
                    'content_graph': item['content_graph'],
                    'comments_graph': [comment_graph]
                }
            )

        content_graph = []
        comments_graph = []
        comments_batch = []
        labels = []
        for index, item in enumerate(batch):
            labels.append(item['label'])
            content_x, content_edge_index = item['content_graph']
            content_x = torch.tensor(content_x, dtype=torch.long)
            content_edge_index = torch.tensor(content_edge_index, dtype=torch.long)
            content_graph.append(Data(x=content_x, edge_index=content_edge_index).to(self.device))
            for comment in item['comments_graph']:
                comments_batch.append(index)
                comment_x, comment_edge_index = comment
                comment_x = torch.tensor(comment_x, dtype=torch.long)
                comment_edge_index = torch.tensor(comment_edge_index, dtype=torch.long)
                comments_graph.append(Data(x=comment_x, edge_index=comment_edge_index).to(self.device))
        content_graph = Batch.from_data_list(content_graph)
        comments_graph = Batch.from_data_list(comments_graph)
        labels = torch.tensor(labels).to(self.device)
        comments_batch = torch.tensor(comments_batch, dtype=torch.long)
        return {
            'content_graph': content_graph,
            'comments_graph': comments_graph,
            'comments_batch': comments_batch,
            'label': labels,
            'batch_size': len(batch)
        }


def main():
    dataset = NewsDataset('politifact', 'vanilla', 'cuda')
    for item in dataset:
        print(item)
        input()



if __name__ == '__main__':
    main()

import random
import torch
import json
import os
from argparse import ArgumentParser
from polluted_dataset import MyDataset, MySampler
from torch.utils.data import DataLoader
from model import Hyphen
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import random


device = torch.device('cuda')


parser = ArgumentParser()
parser.add_argument('--pollution', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

pollution = args.pollution
dataset_name = args.dataset


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred, average='macro') * 100


def forward_one_batch(model, batch):
    content, comment, label, subgraphs = batch['content'], batch['comment'], batch['label'], batch['subgraphs']
    return model(content, comment, subgraphs, label)


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in loader:
        out, loss = forward_one_batch(model, batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    metric = get_metric(all_truth, all_preds)
    return metric


def updating(fold_index, dataset):
    train_indices = []
    test_indices = []
    for _ in range(10):
        index = json.load(open(f'../../datasets/fold/{dataset_name}_{_}.json'))
        if fold_index == _:
            test_indices += index
        else:
            train_indices += index
    random.seed(20240901)
    random.shuffle(train_indices)
    train_sampler = MySampler(train_indices, shuffle=False)
    test_sampler = MySampler(test_indices, shuffle=False)

    size = len(train_indices) // 100
    size = max(size // 5 * 5, 5)

    train_loader = DataLoader(dataset, batch_size=5,
                              collate_fn=dataset.get_collate_fn(device), sampler=train_sampler)
    test_loader = DataLoader(dataset, batch_size=16,
                             collate_fn=dataset.get_collate_fn(device), sampler=test_sampler)
    word_embeddings = train_loader.dataset.word_embeddings.numpy()
    model = Hyphen(embedding_matrix=word_embeddings,
                   word_hidden_size=256,
                   sent_hidden_size=128,
                   device=device,
                   graph_hidden=256,
                   num_classes=train_loader.dataset.num_class,
                   latent_dim=256,
                   graph_glove_dim=300,
                   ).to(device)
    checkpoint_dir = f'checkpoints/{dataset_name}_{fold_index}'
    checkpoint_file = sorted(os.listdir(checkpoint_dir))[0]
    # print(checkpoint_file)
    checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}')
    model.load_state_dict(checkpoint)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    cnt = 0
    trained_size = 0
    for batch in train_loader:
        model.train()
        optimizer.zero_grad()
        out, loss = forward_one_batch(model, batch)
        loss.backward()
        optimizer.step()
        trained_size += 5
        if trained_size % size == 0:
            metric = inference(model, test_loader)
            print(metric)
            cnt += 1
            if cnt == 10:
                break
    print('-------------')


def main():
    print(dataset_name, pollution)
    dataset = MyDataset(dataset_name, pollution)
    for _ in range(10):
        updating(_, dataset)


if __name__ == '__main__':
    main()

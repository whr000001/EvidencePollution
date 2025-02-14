import torch
import json
import os
from argparse import ArgumentParser
from polluted_dataset import NewsDataset, my_collate_fn, MySampler
from torch.utils.data import DataLoader
from model import MyModel
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


device = torch.device('cuda')


parser = ArgumentParser()
parser.add_argument('--pollution', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

pollution = args.pollution
dataset_name = args.dataset


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred, average='macro') * 100


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    for batch in tqdm(loader, leave=False):
        out, loss = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    metric = get_metric(all_truth, all_preds)
    return metric


def main():
    dataset = NewsDataset(dataset_name, pollution)
    acc, f1 = [], []
    for _ in range(10):
        fold_index = json.load(open(f'../../datasets_sampled/fold/{dataset_name}_{_}.json'))
        sampler = MySampler(fold_index, shuffle=False)
        loader = DataLoader(dataset, batch_size=8,
                            collate_fn=lambda x: my_collate_fn(x, device), sampler=sampler)
        model = MyModel(num_class=loader.dataset.num_class, device=device).to(device)
        checkpoint_dir = f'checkpoints/{dataset_name}_{_}'
        checkpoint_file = sorted(os.listdir(checkpoint_dir))[0]
        # print(checkpoint_file)
        checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}')
        model.load_state_dict(checkpoint)
        metric = inference(model, loader)
        acc.append(metric[0])
        f1.append(metric[1])
    print(dataset_name, pollution)
    print('---')
    for item in acc:
        print(f'{item:.2f}')
    print('---')
    # print('----------------------------')
    for item in f1:
        print(f'{item:.2f}')
    print('---')


if __name__ == '__main__':
    main()

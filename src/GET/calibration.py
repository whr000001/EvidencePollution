import torch
from argparse import ArgumentParser
from dataset import NewsDataset, my_collate_fn, MySampler
from torch.utils.data import DataLoader
import json
from model import MyModel
from sklearn.metrics import f1_score, accuracy_score
import os
from tqdm import tqdm


device = torch.device('cuda')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred, average='macro') * 100


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_likelihoods = []
    for batch in tqdm(loader, leave=False):
        out, loss = model(batch)
        likelihoods = torch.softmax(out, dim=-1).to('cpu')

        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_likelihoods.append(likelihoods)
    all_likelihoods = torch.cat(all_likelihoods, dim=0).numpy().tolist()
    all_truth = torch.cat(all_truth, dim=0).numpy().tolist()
    return all_likelihoods, all_truth


def main():
    dataset = NewsDataset(dataset_name)
    likelihoods = []
    truth = []
    for _ in range(10):
        fold_index = json.load(open(f'../../datasets_sampled/fold/{dataset_name}_{_}.json'))
        sampler = MySampler(fold_index, shuffle=True)
        loader = DataLoader(dataset, batch_size=5,
                            collate_fn=lambda x: my_collate_fn(x, device), sampler=sampler)
        model = MyModel(num_class=loader.dataset.num_class, device=device).to(device)
        checkpoint_dir = f'checkpoints/{dataset_name}_{_}'
        checkpoint_file = sorted(os.listdir(checkpoint_dir))[0]
        # print(checkpoint_file)
        checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}')
        model.load_state_dict(checkpoint)
        likelihood, label = inference(model, loader)
        likelihoods += likelihood
        truth += label
    json.dump([likelihoods, truth], open(f'likelihoods/{dataset_name}.json', 'w'))


if __name__ == '__main__':
    main()

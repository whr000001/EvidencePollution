import torch
from argparse import ArgumentParser
from dataset import DefendDataset, MySampler, my_collate_fn
from torch.utils.data import DataLoader
import json
from model import DEFEND
from sklearn.metrics import f1_score, accuracy_score
import os
from tqdm import tqdm


device = torch.device('cuda')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset


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
    dataset = DefendDataset(dataset_name)
    likelihoods = []
    truth = []
    for _ in range(10):
        fold_index = json.load(open(f'../../datasets_sampled/fold/{dataset_name}_{_}.json'))
        sampler = MySampler(fold_index, shuffle=False)
        loader = DataLoader(dataset, batch_size=16,
                            collate_fn=my_collate_fn, sampler=sampler)
        model = DEFEND(embedding_dim=300, hidden_dim=256, k=80,
                       num_classes=loader.dataset.num_class).to(device)
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

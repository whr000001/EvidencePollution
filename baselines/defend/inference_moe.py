import torch
import json
import os
from argparse import ArgumentParser
from dataset_moe import DefendDataset, MySampler
from torch.utils.data import DataLoader
from model import DEFEND
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
def inference(model, dataset, indices):
    model.eval()
    all_truth = []
    all_preds = []
    for index in tqdm(indices, leave=False):
        batch = dataset[index]
        out, loss = model(batch)
        preds = out.argmax(-1)
        pred, _ = torch.mode(preds)
        truth = batch['label'][0]
        all_truth.append(truth.to('cpu'))
        all_preds.append(pred.to('cpu'))
    all_preds = torch.stack(all_preds, dim=0).numpy()
    all_truth = torch.stack(all_truth, dim=0).numpy()
    metric = get_metric(all_truth, all_preds)
    return metric


def main():
    if pollution is None:
        data_path = f'../../datasets/{dataset_name}.json'
    else:
        data_path = f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'
    dataset = DefendDataset(data_path, device)
    acc, f1 = [], []
    for _ in range(10):
        fold_index = json.load(open(f'../../datasets/fold/{dataset_name}_{_}.json'))
        model = DEFEND(embedding_dim=300, hidden_dim=256, k=80,
                       num_classes=dataset.num_class).to(device)
        checkpoint_dir = f'checkpoints/{dataset_name}_{_}'
        checkpoint_file = sorted(os.listdir(checkpoint_dir))[0]
        # print(checkpoint_file)
        checkpoint = torch.load(f'{checkpoint_dir}/{checkpoint_file}')
        model.load_state_dict(checkpoint)
        metric = inference(model, dataset, fold_index)
        acc.append(metric[0])
        f1.append(metric[1])
    print(acc, f1)


if __name__ == '__main__':
    main()

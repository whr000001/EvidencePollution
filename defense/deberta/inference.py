import json
from dataset import MyDataset
from model import MyModel
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--val_set', type=str)
parser.add_argument('--train_set', type=str)
args = parser.parse_args()

val_set_name = args.val_set
train_set_name = args.train_set
batch_size = 32


@torch.no_grad()
def inference(model, loader):
    model.eval()
    all_truth = []
    all_preds = []
    all_score = []
    for batch in loader:
        out, _ = model(batch)
        score = torch.softmax(out, dim=-1)
        score = score[:, 1].to('cpu')
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
        all_score.append(score)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    all_score = torch.cat(all_score, dim=0).numpy()
    x = roc_auc_score(all_truth, all_score) * 100
    y = f1_score(all_truth, all_preds) * 100
    return x, y, accuracy_score(all_truth, all_preds) * 100, precision_score(all_truth, all_preds) * 100, recall_score(all_truth, all_preds) * 100
    # print('auc: {:.2f}'.format())
    # print('f1: {:.2f}'.format())
    # print('auc: ', roc_auc_score(all_truth, all_score))
    # print(accuracy_score(all_truth, all_preds))


def main():
    pollutions = ['vanilla', 'support', 'deny', 'as_publisher', 'echo', 'makeup', 'spread',
                  'rephrase', 'rewrite', 'reverse', 'modify']
    for x in tqdm(pollutions):
        save_path = f'checkpoints/{x}'
        checkpoint_names = os.listdir(save_path)
        checkpoint_names = sorted(checkpoint_names, reverse=True)
        checkpoint_name = checkpoint_names[0]
        # print(checkpoint_name)
        checkpoint = torch.load(f'{save_path}/{checkpoint_name}')
        model = MyModel().to(device)
        model.load_state_dict(checkpoint)
        for y in pollutions:
            test_set = MyDataset(y, 'test')
            test_loader = DataLoader(test_set, batch_size=batch_size,
                                     collate_fn=test_set.get_collate_fn(device), shuffle=True)
            out = inference(model, test_loader)
            print(out)


if __name__ == '__main__':
    main()

import json
from dataset import MyDataset
from model import MyModel
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from argparse import ArgumentParser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--run', type=int, default=3)
args = parser.parse_args()

dataset_name = args.dataset
batch_size = args.batch_size
lr = args.lr


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100


def train_one_epoch(model, optimizer, loader, epoch):
    model.train()
    ave_loss = 0
    cnt = 0
    all_truth = []
    all_preds = []
    # pbar = tqdm(loader, desc='train {} epoch'.format(epoch), leave=False)
    for batch in loader:
        optimizer.zero_grad()
        out, loss = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        loss.backward()
        optimizer.step()
        ave_loss += loss.item() * len(batch)
        cnt += len(batch)
        all_truth.append(truth)
        all_preds.append(preds)
    ave_loss /= cnt
    # print('train loss: {:.4f}'.format(ave_loss))
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return ave_loss, get_metric(all_truth, all_preds)


@torch.no_grad()
def validation(model, loader, epoch):
    model.eval()
    all_truth = []
    all_preds = []
    # pbar = tqdm(loader, desc='valuate {} epoch'.format(epoch), leave=False)
    for batch in loader:
        out, _ = model(batch)
        preds = out.argmax(-1).to('cpu')
        truth = batch['label'].to('cpu')
        all_truth.append(truth)
        all_preds.append(preds)
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_truth = torch.cat(all_truth, dim=0).numpy()
    return get_metric(all_truth, all_preds)


def train(train_loader, val_loader, test_loader, name):
    save_path = 'checkpoints/{}'.format(name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if len(os.listdir(save_path)) == 0:
        history_metrics = 0
    else:
        history_metrics = float(os.listdir(save_path)[0].replace('.pt', ''))

    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_acc = 0
    best_state = model.state_dict()
    for key, value in best_state.items():
        best_state[key] = value.clone()
    pbar = tqdm(range(5), leave=False, desc=name)
    for _ in pbar:
        train_loss, train_acc = train_one_epoch(model, optimizer, train_loader, _)
        acc = validation(model, val_loader, _)
        if acc > best_acc:
            best_acc = acc
            for key, value in model.state_dict().items():
                best_state[key] = value.clone()
        pbar.set_postfix_str('train loss: {:.2f}, train metric: {:.2f}, now best val metric: {:.2f}'.
                             format(train_loss, train_acc, best_acc))
        # print('train loss: {:.2f}, train metric: {:.2f}, now best val metric: {:.2f}'.
        #       format(train_loss, train_acc, best_acc))
    model.load_state_dict(best_state)
    acc = validation(model, test_loader, 0)
    print('train {} done. val metric: {:.2f}, test metric: {:.2f}'.format(name, best_acc, acc))

    if acc > history_metrics:
        if os.path.exists(f'{save_path}/{history_metrics:.2f}.pt'):
            os.remove(f'{save_path}/{history_metrics:.2f}.pt')
        torch.save(best_state, f'{save_path}/{acc:.2f}.pt')


def main():
    # model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
    train_set = MyDataset(dataset_name, 'train')
    val_set = MyDataset(dataset_name, 'val')
    test_set = MyDataset(dataset_name, 'test')

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              collate_fn=train_set.get_collate_fn(device), shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            collate_fn=train_set.get_collate_fn(device), shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size,
                             collate_fn=train_set.get_collate_fn(device), shuffle=False)
    train_name = f'{dataset_name}'
    for _ in range(args.run):
        train(train_loader, val_loader, test_loader, train_name)


if __name__ == '__main__':
    main()

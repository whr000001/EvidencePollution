import json
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--pollution', type=str)
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

pollution = args.pollution
dataset_name = args.dataset


def get_metric(y_true, y_pred):
    return accuracy_score(y_true, y_pred) * 100, f1_score(y_true, y_pred, average='macro') * 100


def get_prediction(data, dataset):
    if dataset in ['politifact', 'gossipcop', 'antivax']:
        real_id = data.find('real')
        fake_id = data.find('fake')
        if real_id != -1 and fake_id == -1:
            return 0
        elif real_id == -1 and fake_id != -1:
            return 1
        elif real_id != -1 and fake_id != -1:
            return int(fake_id < real_id)
        else:
            return 0
    elif dataset in ['hasoc']:
        hate_id = data.find('hate')
        not_id = data.find('not')
        if not_id != -1 and hate_id == -1:
            return 0
        elif not_id == -1 and hate_id != -1:
            return 1
        elif not_id != -1 and hate_id != -1:
            return int(hate_id < not_id)
        else:
            return 0
    elif dataset in ['pheme', 'twitter15', 'twitter16']:
        rumor_id = data.find('rumor')
        not_id = data.find('not')
        if not_id != -1 and rumor_id == -1:
            return 0
        elif not_id == -1 and rumor_id != -1:
            return 1
        elif not_id != -1 and rumor_id != -1:
            return int(rumor_id < not_id)
        else:
            return 0
    elif dataset in ['rumoureval']:
        rumor_id = data.find('rumor')
        not_id = data.find('not a')
        unverified_id = data.find('verified')
        rumor_id = 10000000 if rumor_id == -1 else rumor_id
        not_id = 10000000 if not_id == -1 else not_id
        unverified_id = 10000000 if unverified_id == -1 else unverified_id
        val = [not_id, rumor_id, unverified_id]
        if val == [10000000, 10000000, 10000000]:
            return 2
        ans = np.argmin(val)
        return ans
    elif dataset in ['figlang_reddit', 'figlang_twitter']:
        sarcastic_id = data.find('sarcastic')
        not_id = data.find('not')
        if not_id != -1 and sarcastic_id == -1:
            return 0
        elif not_id == -1 and sarcastic_id != -1:
            return 1
        elif not_id != -1 and sarcastic_id != -1:
            return int(sarcastic_id < not_id)
        else:
            return 0


def main():
    file_name = f'{dataset_name}_{pollution}'

    results = json.load(open(f'res/{file_name}.json'))
    data = json.load(open(f'../../datasets/{dataset_name}.json'))
    all_label = []
    all_preds = []
    for item, res in zip(data, results):
        label = item['label']
        res = res[0]
        prediction = get_prediction(res, dataset_name)
        all_label.append(label)
        all_preds.append(prediction)
    all_preds = np.array(all_preds)
    all_label = np.array(all_label)
    acc, f1 = [], []
    for _ in range(10):
        fold_index = json.load(open(f'../../datasets/fold/{dataset_name}_{_}.json'))
        metric = get_metric(all_label[fold_index], all_preds[fold_index])
        acc.append(metric[0])
        f1.append(metric[1])
    print(acc)
    print(f1)


if __name__ == '__main__':
    main()

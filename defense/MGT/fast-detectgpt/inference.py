import json
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def main():
    pollutions = ['vanilla', 'publisher', 'echo', 'support', 'oppose', 'makeup', 'amplify',
                  'rephrase', 'rewrite', 'modify', 'reverse']
    for pol in pollutions:
        data = json.load(open(f'results/{pol}.json'))
        human = json.load(open('results/human.json'))
        score = np.array(data + human)
        truth = np.array([1] * 1000 + [0] * 1000)
        preds = np.array([int(item > 0.5) for item in score])
        x = roc_auc_score(truth, score) * 100
        y = f1_score(truth, preds) * 100
        print('Pollution {}: {:.2f} {:.2f}'.format(pol, x, y))


if __name__ == '__main__':
    main()

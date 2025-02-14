import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score, precision_score


def main():
    pollutions = ['vanilla', 'support', 'deny', 'as_publisher', 'echo', 'makeup', 'spread',
                  'rephrase', 'rewrite', 'reverse', 'modify']
    for pol in pollutions:
        data = json.load(open(f'results/{pol}.json'))
        human = json.load(open('results/human.json'))
        score = np.array(data + human)
        truth = np.array([1] * 1000 + [0] * 1000)
        preds = np.array([int(item > 0.5) for item in score])
        x = roc_auc_score(truth, score) * 100
        y = f1_score(truth, preds) * 100
        print(f'{pol}: {x:.2f}&{y:.2f}')
        a = accuracy_score(truth, preds) * 100
        b = precision_score(truth, preds) * 100
        c = recall_score(truth, preds) * 100
        # print(x, y, a, b, c)



if __name__ == '__main__':
    main()

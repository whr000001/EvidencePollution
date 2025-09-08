import os
import json
from detector import Binoculars
from tqdm import tqdm


def main():
    path = '../datasets'
    files = os.listdir(path)

    # pls enter your own path here
    model = Binoculars(observer_name_or_path='tiiuae/falcon-7b',
                       performer_name_or_path='tiiuae/falcon-7b-instruct')
    for file in files:
        data = json.load(open(f'../datasets/{file}'))
        data = data[1000:]
        # the last half of the elements are used for testing (the first half is used to train deberta)
        out = []
        for item in tqdm(data, desc=f'{file}', leave=False):
            pred = model.predict(item)
            score = model.compute_score(item)
            out.append([pred, score])
        json.dump(out, open(f'results/{file}', 'w'))


if __name__ == '__main__':
    main()

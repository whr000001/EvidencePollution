import os


def main():
    datasets = ['politifact', 'gossipcop', 'antivax', 'hasoc', 'twitter15',
                'twitter16', 'pheme', 'rumoureval', 'figlang_twitter', 'figlang_reddit']
    for dataset in datasets[5:]:
        cmd = f'python train_one_dataset.py --dataset {dataset} --gpu 7'
        os.system(cmd)


if __name__ == '__main__':
    main()

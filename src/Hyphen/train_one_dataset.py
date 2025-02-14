import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu', type=int)
args = parser.parse_args()

parameters = {
    'politifact': (1e-3, 32),
    'gossipcop': (1e-3, 32),
    'antivax': (1e-3, 32),
    'hasoc': (1e-3, 32),
    'pheme': (1e-3, 32),
    'twitter15': (1e-3, 32),
    'twitter16': (1e-3, 32),
    'rumoureval': (1e-3, 32),
    'figlang_twitter': (1e-3, 32),
    'figlang_reddit': (1e-3, 32)
}


def main():
    lr, batch_size = parameters[args.dataset]
    for _ in range(10):
        cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python train.py --dataset {args.dataset} --run 5 --fold {_}'
        cmd += f' --lr {lr}'
        cmd += f' --batch_size {batch_size}'
        os.system(cmd)
        # print(cmd)


if __name__ == '__main__':
    main()

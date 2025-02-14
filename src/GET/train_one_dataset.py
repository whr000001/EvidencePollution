import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu', type=int)
args = parser.parse_args()


def main():
    for _ in range(10):
        cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python train.py --dataset {args.dataset} --run 5 --fold {_}'
        os.system(cmd)


if __name__ == '__main__':
    main()

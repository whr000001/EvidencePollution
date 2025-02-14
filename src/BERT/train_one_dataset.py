import os
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--gpu', type=int)
args = parser.parse_args()


def main():
    batch_size = 16
    for _ in range(10):
        cmd = f'CUDA_VISIBLE_DEVICES={args.gpu} python train_without_comments.py --dataset {args.dataset} --run 5 --fold {_}'
        cmd += f' --batch_size {batch_size}'
        os.system(cmd)


if __name__ == '__main__':
    main()

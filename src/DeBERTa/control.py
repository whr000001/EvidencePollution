import os


def main():
    # os.system('python train_one_dataset_1.py --dataset politifact --gpu 1')
    # os.system('python train_one_dataset_1.py --dataset gossipcop --gpu 1')
    # os.system('python train_one_dataset_1.py --dataset antivax --gpu 1')
    # os.system('python train_one_dataset_1.py --dataset pheme --gpu 1')
    # os.system('python train_one_dataset_1.py --dataset twitter15 --gpu 1')
    # os.system('python train_one_dataset_1.py --dataset twitter16 --gpu 1')
    os.system('python train_one_dataset_1.py --dataset rumoureval --gpu 0 &')
    os.system('python train_one_dataset_1.py --dataset hasoc --gpu 5 &')
    os.system('python train_one_dataset_1.py --dataset figlang_twitter --gpu 6 &')
    os.system('python train_one_dataset_1.py --dataset figlang_reddit --gpu 7 &')


if __name__ == '__main__':
    main()

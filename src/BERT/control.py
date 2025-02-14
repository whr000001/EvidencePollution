import os


def main():
    # os.system('python train_one_dataset_1.py --dataset politifact --gpu 5')
    os.system('python train_one_dataset.py --dataset gossipcop --gpu 4 &')
    os.system('python train_one_dataset.py --dataset antivax --gpu 5 &')
    os.system('python train_one_dataset.py --dataset pheme --gpu 6 &')
    os.system('python train_one_dataset.py --dataset twitter15 --gpu 7 &')
    os.system('python train_one_dataset.py --dataset twitter16 --gpu 7 &')
    os.system('python train_one_dataset.py --dataset rumoureval --gpu 5 &')
    os.system('python train_one_dataset.py --dataset hasoc --gpu 6 &')
    os.system('python train_one_dataset.py --dataset figlang_twitter --gpu 4 &')
    os.system('python train_one_dataset.py --dataset figlang_reddit --gpu 7 &')


if __name__ == '__main__':
    main()

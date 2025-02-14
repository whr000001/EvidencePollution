import os


def main():
    # pollutions = ['vanilla', 'as_publisher', 'echo', 'support', 'deny', 'makeup', 'spread',
    #               'rephrase', 'rewrite', 'modify', 'reverse']
    # cmd = f'CUDA_VISIBLE_DEVICES=1 python train.py --dataset {pollutions[8]} &'
    # os.system(cmd)
    # cmd = f'CUDA_VISIBLE_DEVICES=2 python train.py --dataset {pollutions[9]} &'
    # os.system(cmd)
    # cmd = f'CUDA_VISIBLE_DEVICES=3 python train.py --dataset {pollutions[10]} &'
    # os.system(cmd)
    # # cmd = f'CUDA_VISIBLE_DEVICES=4 python train.py --dataset {pollutions[4]} &'
    # # os.system(cmd)
    # # cmd = f'CUDA_VISIBLE_DEVICES=5 python train.py --dataset {pollutions[5]} &'
    # # os.system(cmd)
    # # cmd = f'CUDA_VISIBLE_DEVICES=6 python train.py --dataset {pollutions[6]} &'
    # # os.system(cmd)
    # # cmd = f'CUDA_VISIBLE_DEVICES=7 python train.py --dataset {pollutions[7]} &'
    # # os.system(cmd)

    pollutions = ['vanilla', 'support', 'deny', 'as_publisher', 'echo', 'makeup', 'spread',
                  'rephrase', 'rewrite', 'reverse', 'modify']
    for x in pollutions:
        for y in pollutions:
            cmd = f'CUDA_VISIBLE_DEVICES=7 python inference.py --train_set {x} --val_set {y}'
            os.system(cmd)


if __name__ == '__main__':
    main()

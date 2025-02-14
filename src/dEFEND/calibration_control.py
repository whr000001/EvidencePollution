import os


dataset_names = ['politifact', 'gossipcop', 'antivax', 'hasoc', 'pheme', 'twitter15', 'twitter16', 'rumoureval',
                 'figlang_twitter', 'figlang_reddit']
pollutions = ['vanilla', 'as_publisher', 'echo', 'support', 'deny', 'makeup', 'spread',
              'rephrase', 'rewrite', 'modify', 'reverse', 'remove', 'repeat']

for dataset in dataset_names:
    cmd = f'CUDA_VISIBLE_DEVICES=4 python calibration.py ' \
          f'--dataset {dataset}'
    os.system(cmd)
    for pollution in pollutions:
        cmd = f'CUDA_VISIBLE_DEVICES=4 python calibration_pollution.py ' \
              f'--dataset {dataset} --pollution {pollution}'
        os.system(cmd)

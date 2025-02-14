import os


dataset_names = ['politifact', 'gossipcop', 'antivax', 'hasoc', 'pheme', 'twitter15', 'twitter16', 'rumoureval',
                 'figlang_twitter', 'figlang_reddit']
pollutions = ['vanilla', 'as_publisher', 'echo', 'support', 'deny', 'makeup', 'spread',
              'rephrase', 'rewrite', 'modify', 'reverse', 'remove', 'repeat']


for pollution in pollutions:
    for dataset in dataset_names:
        cmd = f'python evaluation_with_pollution.py ' \
              f'--dataset {dataset} --pollution {pollution} >> log.txt'
        os.system(cmd)

import amrlib
import json
import penman
import torch
from amr_coref.coref.inference import Inference
import os
from tqdm import tqdm
from argparse import ArgumentParser
from amr_utils import modify_variables, clean_gen, coreference_edges, concept_merge, get_glove, var2word, to_dict
from penman.models.noop import NoOpModel
import ast
import networkx as nx
import dgl
import pickle
from nltk.tokenize import sent_tokenize
import numpy as np
from argparse import ArgumentParser
from dgl.heterograph import DGLGraph


parser = ArgumentParser()
parser.add_argument('--a', action='store_true')
parser.add_argument('--b', action='store_true')
parser.add_argument('--c', action='store_true')
parser.add_argument('--d', action='store_true')
parser.add_argument('--e', action='store_true')
parser.add_argument('--f', action='store_true')
parser.add_argument('--g', action='store_true')
parser.add_argument('--h', action='store_true')
parser.add_argument('--i', action='store_true')
parser.add_argument('--j', action='store_true')
args = parser.parse_args()
selected = []
if args.a:
    selected.append(0)
if args.b:
    selected.append(1)
if args.c:
    selected.append(2)
if args.d:
    selected.append(3)
if args.e:
    selected.append(4)
if args.f:
    selected.append(5)
if args.g:
    selected.append(6)
if args.h:
    selected.append(7)
if args.i:
    selected.append(8)
if args.j:
    selected.append(9)


os.environ['TOKENIZERS_PARALLELISM'] = 'True'
dataset_list = ['antivax', 'figlang_reddit', 'figlang_twitter', 'gossipcop', 'hasoc', 'pheme',
                'politifact', 'rumoureval', 'twitter15', 'twitter16']


def run_amr_gen():
    stog = amrlib.load_stog_model(model_dir='model_stog')
    # print(stog.parse_sents(['']))
    # input('---------------------------')
    select_list = [dataset_list[_] for _ in selected]
    for dataset in select_list:
        save_dir = f'amr_data/amr_gen/{dataset}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))
        for index in tqdm(range(len(data)), desc=f'{dataset}', leave=False):
            save_path = f'{save_dir}/{index}.json'
            if os.path.exists(save_path):
                continue
            comments = data[index]['comments']
            comments = [' '.join(_.split('\n')) for _ in comments]
            comments = [_.replace('#', ' ') for _ in comments]
            comments = [_.replace('~', ' ') for _ in comments]
            graph = [stog.parse_sents([comments[_]])[0] for _ in range(len(comments))]
            graph = [_ for _ in graph if _ is not None]
            json.dump(graph, open(save_path, 'w'))


def run_amr_var():
    # stog = amrlib.load_stog_model(model_dir='model_stog')
    select_list = [dataset_list[_] for _ in selected]
    for dataset in select_list:
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))

        save_dir = f'amr_data/amr_var/{dataset}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for index in tqdm(range(len(data)), desc=f'{dataset}'):
            save_path = '{}/{}.penman'.format(save_dir, index)
            if os.path.exists(save_path):
                continue
            modified_amr_graph = []
            amr_graph = json.load(open(f'amr_data/amr_gen/{dataset}/{index}.json'))
            for i, amr in enumerate(amr_graph):
                modified_amr = modify_variables(amr, i + 1)
                modified_amr_graph.append(modified_amr)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            penman.dump(modified_amr_graph, save_path, model=NoOpModel())


def run_amr_coref():
    model_dir = 'coref_data/model_coref-v0.1.0/'
    inference = Inference(model_dir)
    select_list = [dataset_list[_] for _ in selected]
    for dataset in select_list:
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))
        save_dir = f'amr_data/amr_coref/{dataset}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for index in tqdm(range(len(data)), desc=f'{dataset}'):
            save_path = '{}/{}.json'.format(save_dir, index)
            if os.path.exists(save_path):
                continue
            graphs = penman.load(f'amr_data/amr_var/{dataset}/{index}.penman', model=NoOpModel())
            try:
                cluster_dict = inference.coreference(graphs)
            except (AssertionError, IndexError, TypeError):
                cluster_dict = {}
            json.dump(cluster_dict, open(save_path, 'w'))


def run_amr_dummy():
    select_list = [dataset_list[_] for _ in selected]
    for dataset in select_list:
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))
        save_dir = f'amr_data/amr_dummy/{dataset}'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        for index in tqdm(range(len(data)), desc=f'{dataset}'):
            save_path = '{}/{}.penman'.format(save_dir, index)
            if os.path.exists(save_path):
                continue
            modified_amr_list = penman.load(f'amr_data/amr_var/{dataset}/{index}.penman', model=NoOpModel())
            instances, edges, attributes = [('d', ':instance', 'dummy')], [], []
            metadata, epidata = {'snt': '', 'lemmas': [], 'tokens': []}, {('d', ':instance', 'dummy'): []}
            subgraph_list = []
            for graph in modified_amr_list:
                node_list = [source for source, _, _ in graph.instances()]
                subgraph_list.append(node_list)

                edges.append(('d', ':COMMENT', graph.top))
                instances.extend(graph.instances())
                edges.extend(graph.edges())
                attributes.extend(graph.attributes())
                metadata['snt'] += "{} ".format(graph.metadata['snt'])
                metadata['lemmas'].extend(ast.literal_eval(graph.metadata['lemmas']))
                metadata['tokens'].extend(ast.literal_eval(graph.metadata['tokens']))
                epidata[('d', ':COMMENT', graph.top)] = [penman.layout.Push(graph.top)]
                epidata.update(graph.epidata)
            metadata['tokens'] = json.dumps(metadata['tokens'])
            metadata['lemmas'] = json.dumps(metadata['lemmas'])
            modified = penman.Graph(instances + edges + attributes)
            modified.metadata = metadata
            modified.epidata = epidata

            amr_coref = json.load(open(f'amr_data/amr_coref/{dataset}/{index}.json'))

            modified = coreference_edges(modified, amr_coref)

            try:
                modified = concept_merge(modified)
            except Exception as e:
                print(e)
                pass
            modified.metadata['subgraphs'] = json.dumps(subgraph_list)
            penman.dump([modified], save_path, model=NoOpModel())


def run_amr_dgl():
    glove = get_glove('../../data/glove.6B.300d.txt')
    select_list = [dataset_list[_] for _ in selected]
    for dataset in select_list:
        save_dir = f'amr_data/amr_dgl/{dataset}.pkl'
        # if os.path.exists(save_dir):
        #     continue
        data = json.load(open(f'../../datasets_sampled/{dataset}.json'))
        output = []
        for index in tqdm(range(len(data)), desc=f'{dataset}'):
            p_graph = penman.load(f'amr_data/amr_dummy/{dataset}/{index}.penman', model=NoOpModel())[0]
            v2w = var2word(p_graph)
            nx_graph = nx.MultiDiGraph()
            nx_graph.add_edges_from([(s, t) for s, _, t in p_graph.edges()])
            if nx_graph.number_of_nodes() == 0:
                nx_graph.add_nodes_from(['d'])
            temp = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted', label_attribute='original')
            original2new = {temp.nodes[i]['original']: i for i in temp.nodes}
            subgraphs = [[original2new[j] for j in i] for i in json.loads(p_graph.metadata['subgraphs'])]
            if len(subgraphs) == 0:
                subgraphs = [[original2new['d']]]
            MAP = {i: glove.get(v2w[i], [0] * 300) for i in nx_graph.nodes()}
            attr = to_dict(MAP)
            nx.set_node_attributes(nx_graph, attr)
            dgl_graph = dgl.from_networkx(nx_graph, node_attrs=['feat'])
            label = data[index]['label']
            content = data[index]['content']

            sample = {'label': label, 'graph': dgl_graph, 'content': content, 'id': index, 'subgraphs': subgraphs}
            output.append(sample)
        print(len(output))
        with open(save_dir, 'wb') as f:
            pickle.dump(output, f)


def save_glove():
    if os.path.exists('../../data/word_list.json') and os.path.exists('../../data/word_vecs.pt'):
        return
    glove_dir = '../../data/glove.6B.300d.txt'
    word_list = []
    word_vecs = []
    with open(glove_dir) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = values[1:]
            coefs = [float(_) for _ in coefs]
            word_list.append(word)
            word_vecs.append(coefs)
    word_vecs = torch.tensor(word_vecs, dtype=torch.float)
    json.dump(word_list, open('../../data/word_list.json', 'w'))
    torch.save(word_vecs, '../../data/word_vecs.pt')


def main():
    run_amr_gen()
    run_amr_var()
    run_amr_coref()
    run_amr_dummy()
    run_amr_dgl()
    save_glove()


if __name__ == '__main__':
    main()

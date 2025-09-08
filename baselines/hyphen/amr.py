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

os.environ['TOKENIZERS_PARALLELISM'] = 'True'

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--pollution', type=str)
args = parser.parse_args()

dataset_name = args.dataset
pollution = args.pollution


def run_amr_gen():
    stog = amrlib.load_stog_model(model_dir='model_stog')

    save_dir = f'amr_data/amr_gen/{dataset_name}_{pollution}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))
    for index in tqdm(range(len(data)), desc=save_dir, leave=False):
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
    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))
    save_dir = f'amr_data/amr_var/{dataset_name}_{pollution}'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for index in tqdm(range(len(data)), desc=save_dir):
        save_path = '{}/{}.penman'.format(save_dir, index)
        if os.path.exists(save_path):
            continue
        modified_amr_graph = []
        amr_graph = json.load(open(f'amr_data/amr_gen/{dataset_name}_{pollution}/{index}.json'))
        for i, amr in enumerate(amr_graph):
            modified_amr = modify_variables(amr, i + 1)
            modified_amr_graph.append(modified_amr)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        penman.dump(modified_amr_graph, save_path, model=NoOpModel())


def run_amr_coref():
    model_dir = 'coref_data/model_coref-v0.1.0/'
    inference = Inference(model_dir)

    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))

    save_dir = f'amr_data/amr_coref/{dataset_name}_{pollution}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for index in tqdm(range(len(data)), desc=save_dir):
        save_path = '{}/{}.json'.format(save_dir, index)
        if os.path.exists(save_path):
            continue
        graphs = penman.load(f'amr_data/amr_var/{dataset_name}_{pollution}/{index}.penman', model=NoOpModel())
        try:
            cluster_dict = inference.coreference(graphs)
        except (AssertionError, IndexError, TypeError):
            cluster_dict = {}
        json.dump(cluster_dict, open(save_path, 'w'))


def run_amr_dummy():
    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))
    save_dir = f'amr_data/amr_dummy/{dataset_name}_{pollution}'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for index in tqdm(range(len(data)), desc=save_dir):
        save_path = '{}/{}.penman'.format(save_dir, index)
        if os.path.exists(save_path):
            continue
        modified_amr_list = penman.load(f'amr_data/amr_var/{dataset_name}_{pollution}/{index}.penman', model=NoOpModel())
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

        amr_coref = json.load(open(f'amr_data/amr_coref/{dataset_name}_{pollution}/{index}.json'))

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

    save_dir = f'amr_data/amr_dgl/{dataset_name}_{pollution}.pkl'
    # if os.path.exists(save_dir):
    #     continue
    if pollution is None:
        data = json.load(open(f'../../datasets/{dataset_name}.json'))
    else:
        data = json.load(open(f'../../attack/polluted_datasets/{dataset_name}_{pollution}.json'))
    output = []
    for index in tqdm(range(len(data)), desc=save_dir):
        p_graph = penman.load(f'amr_data/amr_dummy/{dataset_name}_{pollution}/{index}.penman', model=NoOpModel())[0]
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
    if os.path.exists('../data/word_list.json') and os.path.exists('../data/word_vecs.pt'):
        return
    glove_dir = '../data/glove.6B.300d.txt'
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
    json.dump(word_list, open('../data/word_list.json', 'w'))
    torch.save(word_vecs, '../data/word_vecs.pt')


def main():
    save_glove()
    run_amr_gen()
    run_amr_var()
    run_amr_coref()
    run_amr_dummy()
    run_amr_dgl()


if __name__ == '__main__':
    main()

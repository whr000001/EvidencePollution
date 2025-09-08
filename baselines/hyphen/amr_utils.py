import penman
import json
import os
from amrlib.graph_processing.annotator import add_lemmas
from tqdm import tqdm
import numpy as np


max_comments = 50


def clean_gen(dataset):
    cnt = 0
    data = json.load(open(f'../../datasets/{dataset}.json'))
    for index in tqdm(range(len(data))):
        amr_graph = json.load(open(f'amr_data/amr_gen/{dataset}/{index}.json'))
        if len(amr_graph) > max_comments:
            amr_graph = amr_graph[:max_comments]
            json.dump(amr_graph, open(f'amr_data/amr_gen/{dataset}/{index}.json', 'w'))
            if os.path.exists(f'amr_data/amr_var/{dataset}/{index}.penman'):
                os.remove(f'amr_data/amr_var/{dataset}/{index}.penman')
            if os.path.exists(f'amr_data/amr_coref/{dataset}/{index}.json'):
                os.remove(f'amr_data/amr_coref/{dataset}/{index}.json')
            cnt += 1
        try:
            for amr in amr_graph:
                g = penman.decode(amr)
        except Exception as e:
            print(e)
            print(dataset, index)
            os.remove(f'amr_data/amr_gen/{dataset}/{index}.json')
            cnt += 1
    print(cnt)


def modify_variables(amr, i):
    """This function takes an AMR as input and modifies the variables of the AMR depending on the serial number
    of the AMR. Here, (i) refers to the (i)th comment on a particular news piece.

    Returns: The modified penman Graph of the AMR string.

    Note: This function does not modify the edpidata or metadata of the input AMR. We just modify the variable names
    in this function. Since our ultimate goal is to merge several AMR graphs, it is highly likely that different
    amrs have the same variable names. Thus to distinguish between variables of different amrs we assign unique names
    to different variables."""
    try:
        g = penman.decode(amr)
    except:
        print(amr, i)
        input('---------------------')

    g_meta = add_lemmas(amr, snt_key='snt')  # adding lemmas , tokens to the AMR string

    # create a dictionary for mapping old variables to new variable names

    var, d = list(g.variables()), {}

    for j in range(len(var)):
        d[var[j]] = "c{}-{}".format(i, j)

    # modify the variable names of instances, edges, attributes of the original amr graph
    instances, edges, attributes, epidata = [], [], [], {}
    for source, role, target in g.instances():  # modify the instances
        instances.append((d[source], role, target))
    for source, role, target in g.edges():  # modify the edges
        edges.append((d[source], role, d[target]))

    for source, role, target in g.attributes():  # modify the attributes
        attributes.append((d[source], role, target))

    for (source, role, target) in g.epidata.keys():  # modify the attributes

        push_pop = g.epidata[(source, role, target)]

        modified_epi = []
        for p in push_pop:
            if isinstance(p, penman.layout.Push):
                modified_epi.append(penman.layout.Push(d[p.variable]))
            elif isinstance(p, penman.layout.Pop):
                modified_epi.append(p)
            else:
                print(p)

        # if the epidata key is either an instance or attribute triple
        if (source, role, target) in g.instances() or (source, role, target) in g.attributes():
            epidata[(d[source], role, target)] = modified_epi

        elif (source, role, target) in g.edges():
            epidata[(d[source], role, d[target])] = modified_epi
        else:
            print((source, role, target))

    modified = penman.Graph(instances + edges + attributes)  # return the modifies graph

    modified.metadata = g_meta.metadata  # using the metadata from the original graph

    modified.epidata = epidata  # using the epidata from the original graph -- name changed

    assert len(eval(modified.metadata['lemmas'])) == len(
        eval(modified.metadata['tokens'])), "Length of tokens must be equal to lemmas"

    return modified


def add_edge(graph, source, role, target):
    """Function to add an edge between two previously existing nodes in the graph.

    Here, source and target node instances already exist in "graph" and we simply add an edge with relation "rel"
    between the two. The purpose of this is to add :COREF and :SAME edges

    TODO: Modify the epidata while adding a new edge"""

    edges = [(source, role, target)]  # adding the new edge
    edges.extend(graph.edges())

    # modified amr after adding the required edge
    modified = penman.Graph(graph.instances() + edges + graph.attributes())
    return modified


def coreference_edges(merged_amr, amr_coref, size=None):
    if size is None:
        d = amr_coref
        for relation, cluster in d.items():
            var = [i[1] for i in cluster]
            # amr_coref is sorted according to time (i.e. comments appearing first temporally appear before) by default
            source = var[0]  # the directed edge will start from the comment appearing first: following temporal fashion
            for target in var[1:]:  # add :COREF edges from the source word to all words in the cluster
                added = add_edge(merged_amr, source, ":COREF", target)
                merged_amr = added
        return merged_amr
    d = amr_coref
    for relation, cluster in d.items():
        var = [i[1] for i in cluster]
        source = var[0]  # the directed edge will start from the comment appearing first: following temporal fashion
        source_id = int(source.split('-')[0][1:])
        if source_id > size:
            continue
        for target in var[1:]:  # add :COREF edges from the source word to all words in the cluster
            target_id = int(target.split('-')[0][1:])
            if target_id > size:
                continue
            added = add_edge(merged_amr, source, ":COREF", target)
            merged_amr = added
    return merged_amr


def generate_word2var(normalised_graph):
    """This function returns a word-to-variableNames dictionary.

    A word might be present on multiple nodes. This returns the dictionary storing the nodes for every word."""
    word2var = {}  # a dictionary mapping the words to nodes; example: the word 'name' might belong to 2 nodes

    for (source, role, target) in normalised_graph.instances():
        if target in word2var:
            word2var[target].append(source)
        else:
            word2var[target] = [source]
    return word2var


def normalise_graph(modified):
    """A function to convert the concepts in the amr to meaningful form so that we can apply glove embedding later on
    to find their node representations.

    Note: Removing the part after the hyphen (-) in many of the concept names."""
    normalised_instances = []
    for (source, role, target) in modified.instances():
        if "-" in target:  # for example: "save-01" concept is converted to "save".
            normalised_instances.append((source, role, target[:target.rfind("-")]))
        else:
            normalised_instances.append((source, role, target))
    normalised_graph = penman.Graph(normalised_instances + modified.edges() + modified.attributes())
    return normalised_graph


def concept_merge(modified):
    normalised_graph = normalise_graph(modified)
    word2var = generate_word2var(normalised_graph)
    for word, var in word2var.items():
        head_node = var[0]
        if len(var) > 1:
            for j in var[1:]:
                added = add_edge(normalised_graph, head_node, ":SAME", j)
                normalised_graph = added
    return normalised_graph


def get_glove(g_path):
    glove = {}
    f = open(g_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype=float)
        glove[word] = coefs
    return glove


def var2word(p_graph):
    v2w = {}
    for (source, _, target) in p_graph.instances():
        v2w[source] = target
    return v2w


def to_dict(d):
    return {i: {'feat': j} for i, j in d.items()}

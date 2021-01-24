import pickle
import cv2
import os, glob, sys
import spacy, re
import numpy as np
from gensim.models import FastText as ft
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import unidecode
# nlp = spacy.load('de')
sys.path.append('../utils')
try:
    from conjugate import conjugate_nx
except:
    from data.conjugate import conjugate_nx


def get_all(path, ext="pkl"):
    file_names = glob.glob(os.path.join(path, "*."+ext))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_nodes_ft(nodes, model):
    res = []
    for i, node in enumerate(nodes):
        n = node[:-2]
        # if self.text_length:
        text = node[-2]
        # all_words = proces_text([text], categories=False)[0]
        res_ft = np.zeros((300))
        text = create_text(text)
        try:
            res_ft += model.wv[text]
            # print(text)
        except:
            print("------------->        ", text)
            pass
        # print(all_words)
        n.extend(res_ft)
        res.append(n)
    return res

def isPossibleSequence(s1, s2):
    """
    From TranskribusDU
    :param s1:
    :param s2:
    :return:
    """
    try:
        n1 = int(s1)
        n2 = int(s2)
        return (n1 + 1) == n2 or (n2 + 1) == n1
    except Exception:
        return False

def get_nodes_len(nodes):
    res = []
    for i, node in enumerate(nodes):
        n = node[:-2]
        # if self.text_length:
        text = node[-2]
        n.append(len(text))
        [n.append(x) for x in
            [
                text.isalnum(),
                text.isalpha(),
                text.isdigit(),
            ]
        ]
        res.append(n)
    return res

def get_nodes_notext(nodes):
    res = []
    for i, node in enumerate(nodes):
        n = node[:-2]
        res.append(n)
    return res

def get_edges(edge_feats, edges, node_feats):
    res = []
    for i, node in enumerate(edge_feats):
        ij = edges[i]
        word_i = node_feats[ij[0]][-2]
        word_j = node_feats[ij[1]][-2]
        seq = isPossibleSequence(word_i, word_j)
        n = node[:-1]
        n.append(seq)
        res.append(n)
    return res

def create_text(line):
    """
    ALERT: copied from HTR project, file get_text_lda
    """
    res = ""
    line = line.lower()
    line = unidecode.unidecode(line)
    line =  re.sub(r"[^a-zA-Z0-9 ]+", '', line)
    line = line.lstrip()
    res = line.lower() 
    if res == " " or not res or res == "":
        res = "\""
    return res

def get_nodes_lda(nodes, vectorizer, lda):
    res = []
    for i, node in enumerate(nodes):
        n = node[:-2]
        # if self.text_length:
        text = node[-2]
        X = vectorizer.transform([text])
        topic_results = lda.transform(X)[0]
        # print("\nText: ", text)
        # print("X: ", X)
        # print("topic_results: ", topic_results)
        n.extend(topic_results)
        res.append(n)
    return res

def save_to_file(data, fname):
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

opciones = ["fasttext", "text", "no", "lda"]
opcion = "no"
dir_orig = sys.argv[1]
dir_dest = sys.argv[2]
# dir_orig = ""
# dir_dest =  "/data/READ_ABP_TABLE_ICDAR/icdar_488/all/graphs_preprocessed/graphs_structure_BL_lentext"

fnames = get_all(dir_orig)
print(dir_orig)

# exit()
create_dir(dir_dest)

print("Created {}".format(dir_dest))
c = 0
for fname in fnames:
    c+=1
    f = open(fname, "rb")
    data_load = pickle.load(f)
    f.close()
    fname_xml = fname.split("/")[-1]
    path_to_save = os.path.join(dir_dest, fname_xml)
    print(path_to_save, end=" ")
    nodes = data_load['nodes']
    edges = data_load['edges']
    labels = data_load['labels']
    edge_features = data_load['edge_features']
    ids = data_load['ids']
    data_load['edge_features'] = get_edges(data_load['edge_features'], data_load['edges'], data_load['nodes'])
    if opcion.lower() == opciones[0]:
        pass
        # nodes = get_nodes_ft(nodes, model)
    elif opcion.lower() == opciones[1]:
        nodes = get_nodes_len(nodes)
    elif opcion.lower() == opciones[2]:
        nodes = get_nodes_notext(nodes)
    elif opcion.lower() == opciones[3]:
        pass
        # nodes = get_nodes_lda(nodes, vectorizer, lda)
    data_load['nodes'] = nodes

    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                    nodes, edges, labels, edge_features)
    data_load['conjugate'] = ( ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges)
    save_to_file(data_load, fname=path_to_save)
    print("Done {}/{}".format(c, len(fnames)))

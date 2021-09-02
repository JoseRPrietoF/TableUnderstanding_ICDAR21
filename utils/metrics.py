import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from numpy import zeros, inf, array, argmin
import os
from scipy.optimize import linear_sum_assignment

# os.sysconf()
import networkx as nx, pickle, glob
try:
    from data.conjugate import conjugate_nx
except:
    from conjugate import conjugate_nx


def eval_accuracy(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.where(hyp > 0.5, 1, 0)
    gt = np.where(gt > 0.5, 1, 0)
    acc = []
    p = []
    r = []
    f1 = []
    # accuracy: (tp + tn) / (p + n)
    for i in range(len(gt)):

        accuracy = accuracy_score(hyp[i], gt[i])

        # precision tp / (tp + fp)
        precision = precision_score(hyp[i], gt[i])
        # recall: tp / (tp + fn)
        recall = recall_score(hyp[i], gt[i])
        # f1: 2 tp / (2 tp + fp + fn)
        f1_ = f1_score(hyp[i], gt[i])
        acc.append(accuracy)
        p.append(precision)
        r.append(recall)
        f1.append(f1_)

    return acc, p, r, f1

def eval_graph(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.exp(hyp)
    hyp = np.where(hyp > 0.5, 1, 0)
    # gt = np.where(gt > 0.5, 1, 0)
    # accuracy: (tp + tn) / (p + n)
    # print(gt)
    # print(hyp)
    accuracy = accuracy_score(hyp, gt)

    # precision tp / (tp + fp)
    precision = precision_score(hyp, gt, average='macro')
    # recall: tp / (tp + fn)
    recall = recall_score(hyp, gt, average='macro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1_ = f1_score(hyp, gt, average='macro')
    # print(recall, precision, f1_)
    return accuracy, precision, recall, f1_


def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = nOk / (nOk + nErr + eps)
    fR = nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF

def jaccard_distance(x,y):
    """
        intersection over union
        x and y are of list or set or mixed of
        returns a cost (1-similarity) in [0, 1]
    """
    try:    
        setx = set(x)
        return  1 - (len(setx.intersection(y)) / len(setx.union(y)))
    except ZeroDivisionError:
        return 0.0

def evalHungarian(X,Y,thresh, func_eval=jaccard_distance):
        """
        https://en.wikipedia.org/wiki/Hungarian_algorithm
        """          
        cost = [func_eval(x,y) for x in X for y in Y]
        cost_matrix = np.array(cost, dtype=float).reshape((len(X), len(Y)))
        r1,r2 = linear_sum_assignment(cost_matrix)
        toDel=[]
        for a,i in enumerate(r2):
            # print (r1[a],ri)      
            if 1 - cost_matrix[r1[a],i] < thresh :
                toDel.append(a)                    
        r2 = np.delete(r2,toDel)
        r1 = np.delete(r1,toDel)
        _nOk, _nErr, _nMiss = len(r1), len(X)-len(r1), len(Y)-len(r1)
        return _nOk, _nErr, _nMiss

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def read_results(fname, conjugate=True):
    """
    Since the differents methods tried save results in different formats,
    we try to load all possible formats.
    """
    results = {}
    if conjugate:
        if type(fname) == str: 
            f = open(fname, "r")
            lines = f.readlines()
            f.close()
            for line in lines[1:]:
                id_line, label, prediction = line.split(" ")
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )
        else:
            for id_line, label, prediction in fname:
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = int(label), np.exp(float(prediction))
    else:
        for fname, label, prediction, (i,j) in fname:
            id_line = fname
            results[id_line] = int(label), np.exp(float(prediction))
    return results

def evaluate_graph_IoU(file_list, results, min_w = 0.5, th = 0.8, type_="COL", conjugate=True, all_edges=False, pruned=False):
    ORACLE = False
    type_ = type_.lower()
    results = read_results(results, conjugate=conjugate and not all_edges)
    nOk, nErr, nMiss = 0,0,0
    fP, fR, fF = 0,0,0
    res = []
    fname_search = "52684_002"
    for raw_path in file_list:  
        # if fname_search not in raw_path:
        #     continue
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']   
        if conjugate:
            if data_load.get("conjugate", None) is not None:
                    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
                    # print(nodes)
            else:
                ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
                new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                        nodes, edges, labels, edge_features, idx=None, list_idx=None)
            if len(edges) != len(new_nodes):
                print("Problem with {} {} edges and {} new_nodes".format(raw_path, len(edges), len(new_nodes)))
                #continue
        else:
            ant_nodes = edges
        file_name = raw_path.split("/")[-1].split(".")[0]

        G = nx.Graph()
        G_gt = nx.Graph()
        weighted_edges_GT = []
        weighted_edges = []
        weighted_edges_dict_gt = {}
        out_graph = []

        for i, node in enumerate(nodes):
            # G.add_node(i, attr=node)
            if type_ == "cell":
                r, c = labels[i]['row'], labels[i]['col']
                if r == -1 or c == -1:
                    ctype = -1
                else:
                    ctype = "{}_{}".format(r,c)
            else:
                ctype = labels[i][type_]
            id_ = ids[i]

            if ctype != -1:
                G.add_node(i)
                G_gt.add_node(i)
                cc_type = weighted_edges_dict_gt.get(ctype, set())
                cc_type.add(i)
                weighted_edges_dict_gt[ctype] = cc_type
            else:
                out_graph.append(i)


        aux_dict = {}
        if all_edges:

            for i, node_i in enumerate(nodes):
                for j, node_j in enumerate(nodes):

                    x_coord_o, y_coord_o = node_i[0], node_i[1]
                    x_coord_t, y_coord_t = node_j[0], node_j[1]
                    dif_y = abs(y_coord_o - y_coord_t)
                    dif_x = abs(x_coord_o - x_coord_t)
                    key = "{}_edge{}-{}".format(file_name, i, j)
                    try:
                        gt, hyp_prob = results.get(key)
                    except Exception as e:
                        print("Problem with key {} (no conjugation - all_edges)".format(key))
                        # continue
                        raise e
                    if ORACLE:
                        if i in out_graph or j in out_graph: continue
                        if gt:
                            hyp_prob = 1
                            weighted_edges.append((i,j,hyp_prob))
                        # else:
                        #     hyp_prob = 0
                    else:
                        # if hyp_prob > min_w and dif_x <= 0.01:
                        if hyp_prob > min_w:
                            weighted_edges.append((i,j,hyp_prob))
        elif not all_edges and not conjugate:
            for (i,j) in edges:
                key = "{}_edge{}-{}".format(file_name, i, j)
                key2 = "{}_edge{}-{}".format(file_name, j, i)
                try:
                    gt, hyp_prob = results.get(key, results.get(key2))
                except Exception as e:
                    print("Problem with key {} - {}".format(key, key2))
                    # continue
                    raise e
                if ORACLE:
                        if i in out_graph or j in out_graph: continue
                        if gt:
                            hyp_prob = 1
                            weighted_edges.append((i,j,hyp_prob))
                        # else:
                        #     hyp_prob = 0
                else:
                    # if hyp_prob > min_w and dif_x <= 0.01:
                    if hyp_prob > min_w:
                        weighted_edges.append((i,j,hyp_prob))
        else:
            for idx, (i, j) in enumerate(ant_nodes):
                aux_dict[(i, j)] = idx
            count_acc = []
            for count, (i, j) in enumerate(edges):
                
                # idx_edge = aux_dict.get((i, j), aux_dict.get((j, i)))
                if conjugate:
                    idx_edge = "{}_{}".format(j, i)
                    key = "{}-edge{}".format(file_name, idx_edge)
                    idx_edge2 = "{}_{}".format(i, j)
                    key2 = "{}-edge{}".format(file_name, idx_edge2)
                    if pruned:
                        found = True
                        try:
                            gt, hyp_prob = results.get(key, results.get(key2))
                            # print((j,i), key, key2, "si")
                        except:
                            gt, hyp_prob = 0, 0
                            found = False
                            # print((j,i), key, key2, "no")
                    else:
                        try:
                            gt, hyp_prob = results.get(key, results.get(key2))
                        except Exception as e:
                            print(key, key2)
                            raise e
                else:
                    # print(list(results.keys())[:10], file_name)
                    # exit()
                    key = "{}_edge{}-{}".format(file_name, i, j)
                    key2 = "{}_edge{}-{}".format(file_name, j, i)
                    try:
                        gt, hyp_prob = results.get(key, results.get(key2))
                    except Exception as e:
                        print("Problem with key {} - {}".format(key, key2))
                        # continue
                        raise e
                if ORACLE:
                   
                    if i in out_graph or j in out_graph: continue
                    if gt:
                        hyp_prob = 1
                        weighted_edges.append((i,j,hyp_prob))
                    # else:
                    #     hyp_prob = 0
                else:
                    if hyp_prob > min_w:
                        weighted_edges.append((i,j,hyp_prob))
                # if gt:
                #     weighted_edges_GT.append((i,j,1))
                count_acc.append((hyp_prob > min_w) == gt)

        
        G.add_weighted_edges_from(weighted_edges)
        cc = nx.connected_component_subgraphs(G)
        cc = [sorted(list(c)) for c in cc]

        cc_gt = [ sorted(list(ccs)) for ctype,ccs in weighted_edges_dict_gt.items()]

        cc_gt.sort()
        cc.sort()

        _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, th, jaccard_distance)
        _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)

        res.append([raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF])
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        fP += _fP
        fR += _fR
        fF += _fF
        gt_edges_graph_dict = None
    fP, fR, fF = fP/len(file_list), fR/len(file_list), fF/len(file_list)
    print("_nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {}".format(nOk, nErr, nMiss, fP, fR, fF))
    return fP, fR, fF, res
if __name__ == "__main__":
    test_samples()

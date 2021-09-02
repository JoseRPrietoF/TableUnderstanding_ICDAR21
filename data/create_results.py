import os.path as osp
import os
import glob, pickle
import torch
import numpy as np
# from torch_geometric.utils import grid
import networkx as nx
import logging
import sys
sys.path.append('../utils')
try:
    from conjugate import conjugate_nx
except:
    from data.conjugate import conjugate_nx
try:
    from utils.optparse_graph import Arguments as arguments
except:
    from optparse_graph import Arguments as arguments


def read_fold(p):
    f = open(p)
    lines = f.readlines()
    f.close()
    lines = [x.split(".")[0] for x in lines]
    return lines

def get_all(path, ext="pkl"):
    if path[-1] != "/":
        path = path+"/"
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def main(path, fold_path, dir_fold, type_="cols"):
    conjugate = True
    file_list = get_all(path)
    files_fold = read_fold(fold_path)
    print(len(files_fold))
    f_dest = open(os.path.join(path, "results_{}.txt".format(type_)), "w")
    f_dest.write("ID LABEL PREDICTION\n")
    for raw_path in file_list:
        file_name = raw_path.split("/")[-1].split(".")[0]
        if file_name not in files_fold:
            continue
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        
        ids = data_load['ids']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']

        if conjugate:
            ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
            new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                    nodes, edges, labels, edge_features, idx=None, list_idx=None)
            if len(edges) != len(new_nodes):
                print("Problem with {} {} edges and {} new_nodes".format(raw_path, len(edges), len(new_nodes)))
                #continue
        else:
            ant_nodes = edges

        for i, node in enumerate(new_nodes):
            if type_ == "cols":
                label_i = new_labels_cols[i]
            else:
                label_i = new_labels_rows[i]
            id_ = ids[i]
            f_dest.write("{} {} {}\n".format(id_, label_i, np.log(label_i+0.00000001)))

        # G_gt = nx.Graph()
        # weighted_edges_dict_gt = {}
        # out_graph = []  
        # for i, node in enumerate(nodes):
        #     ctype = labels[i][type_]
        #     id_ = ids[i]
        #     if ctype != -1:
        #         G_gt.add_node(i)
        #         cc_type = weighted_edges_dict_gt.get(ctype, set())
        #         cc_type.add(i)
        #         weighted_edges_dict_gt[ctype] = cc_type
        #     else:
        #         out_graph.append(i)
        # cc_gt = [ sorted(list(ccs)) for ctype,ccs in weighted_edges_dict_gt.items()]
        # cc_gt.sort()
        # print(cc_gt)
        # exit()
    f_dest.close()
if __name__ == "__main__":
    type_="rows"
    path = "/data/READ_ABP_TABLE_ICDAR/icdar_abp_1098/all/graphs_structure_BL_maxBLWidth1.5_multpunct_w2_h1_mult_side_h0_w0_bound0_rows1/"
    # path = "/data/READ_ABP_TABLE_ICDAR/icdar19_abp_small/all/graphs_structure_BL_maxBLWidth1.5_mincell2_multpunct_w1_h1_mult_side_h0_w0_bound0_rows8/"
    # flist_test = "/data2/jose/corpus/tablas_DU/icdar_488/fold1.fold"
    flist_test = "/data2/jose/corpus/tablas_DU/icdar_abp_1098/fold1.fold"
    dir_fold = "/data2/jose/corpus/tablas_DU/icdar_abp_1098/"
    main(path, flist_test, dir_fold, type_)

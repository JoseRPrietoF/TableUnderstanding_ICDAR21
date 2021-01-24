import os.path as osp
import os
import glob, pickle
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.utils import degree
import numpy as np
# from torch_geometric.utils import grid
import networkx as nx
import logging
import sys
from gensim.models import FastText as ft
sys.path.append('../utils')
try:
    from conjugate import conjugate_nx
except:
    from data.conjugate import conjugate_nx
try:
    from utils.optparse_graph import Arguments as arguments
except:
    from optparse_graph import Arguments as arguments

class ABPDataset_BIESO(Dataset):
    """
    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:

    torch_geometric.data.InMemoryDataset.raw_file_names():
    A list of files in the raw_dir which needs to be found in order to skip the download.

    torch_geometric.data.InMemoryDataset.processed_file_names():
    A list of files in the processed_dir which needs to be found in order to skip the processing.

    torch_geometric.data.InMemoryDataset.download():
    Downloads raw data into raw_dir.

    torch_geometric.data.InMemoryDataset.process():
    Processes raw data and saves it into the processed_dir.
    """
    def __init__(self, root, split, flist, opts=None, transform=None, pre_transform=None, onehot = False):
        # super(ABPDataset_BIESO, self).__init__(root, transform, pre_transform, None)
        self.root = None
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = None
        self.__indices__ = None

        self.opts = opts
        self.onehot = onehot
        self.split = split
        self.processed_dir_ = os.path.join(opts.work_dir, split)
        if not os.path.exists(self.processed_dir_):
            os.mkdir(self.processed_dir_)
        self.pre_transform = pre_transform
        self.flist = flist
        self.flist_processed = []
        # self.processed_dir_ = self.path
        self.transform = transform
        self.dict_clases = {
            'B': 0,
            'I': 1,
            'E': 2,
            'S': 3,
            'O': 4
        }
        if opts is not None:
            self.conjugate = self.opts.conjugate
        else:
            self.conjugate = "COL"
        if self.conjugate == "NO":
            self.num_classes = len(self.dict_clases.keys())
        elif self.conjugate == "ALL":
            self.num_classes = 4
        else:
            self.num_classes = 2
        self.ids = []
        self.labels = []

    def get_prob_class(self):
        if self.conjugate != "CR":
            _, counts = np.unique(self.labels, return_counts=True)
            total = counts.sum()
            class_weights = counts / total
        else:
            labels = np.array(self.labels)
            col = labels[:,0]
            row = labels[:,1]
            _, counts = np.unique(col, return_counts=True)
            total = counts.sum()
            class_weights_col = counts / total

            _, counts = np.unique(row, return_counts=True)
            total = counts.sum()
            class_weights_row = counts / total
            return class_weights_col, class_weights_row
        return class_weights

    @property
    def raw_file_names(self):
        # return self.flist
        return []
    @property
    def processed_file_names(self):
        # return self.flist
        return []

    def __len__(self):
        """
        Returns the number of examples in your dataset.
        :return:
        """
        # print(self.processed_file_names)
        return len(self.flist)

    def download(self):
        # Download to `self.raw_dir`.
        # self.raw_paths = []
        print("Download?")

    def add_label_to(self, edges, labels):
        # print(len(edges), len(labels))
        # print(edges)
        for i, info in enumerate(edges):
            x = list(edges[i])
            x.append(labels[i])
            edges[i] = x
        return edges

    def process(self):
        self._process()
    
    def read_results(self, fname):
        results = {}
        if type(fname) == str: 
            f = open(fname, "r")
            lines = f.readlines()
            f.close()
            for line in lines[1:]:
                id_line, label, prediction = line.split(" ")
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )
        else:
            # print(fname)
            for id_line, label, prediction in fname:
                id_line = id_line.split("/")[-1].split(".")[0]
                results[id_line] = int(label), np.exp(float(prediction))
        return results

    def _process(self):
        i = 0
        node_numeration = 0
        # if not os.path.exists(osp.join(self.root, 'processed')):
        #     os.mkdir(osp.join(self.root, 'processed'))
        self.labels = []
        # pna
        # self.deg = torch.zeros(90, dtype=torch.long)
        min_w = 0
        if self.opts.do_prune:
            if self.split == "train":
                path_prune = self.opts.results_prune_tr
            elif self.split == "test":
                path_prune = self.opts.results_prune_te
            results = self.read_results(path_prune)
            min_w = 0.5
        for idx_flist, raw_path in enumerate(self.flist):
            # Read data from `raw_path`.
            file_name = raw_path.split("/")[-1].split(".")[0]
            f = open(raw_path, "rb")
            data_load = pickle.load(f)
            f.close()
            ids = data_load['ids']
            if self.conjugate == "NO":
                labels = []
                for label in data_load['labels']:
                    try:
                        l = self.dict_clases[label['DU_row']]
                    except:
                        l = label
                    labels.append(l)

                # edge_index = torch.tensor(np.array(data_load['edges']).T, dtype=torch.long)
                # edge_index, info_edges = self.get_edge_info(data_load['edges'])
                edge_index = np.array(data_load['edges']).T
                info_edges = data_load['edge_features']
                nodes = data_load['nodes']
                self.labels.extend(labels)
            else:
                nodes = data_load['nodes']
                edges = data_load['edges']
                labels = data_load['labels']
                edge_features = data_load['edge_features']
                            
                if data_load.get("conjugate", None) is not None and not self.opts.do_prune:
                    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
                    # print(nodes)
                else:
                    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                    nodes, edges, labels, edge_features)
                if self.opts.do_prune:
                    # print(len(ant_nodes), len(new_nodes))
                    # print(ant_nodes[:5])
                    # exit()
                    aux_dict = {}
                    edges_pruned = []
                    # for idx, (s, d) in enumerate(edges):
                    #     aux_dict[(s, d)] = idx
                    # for count, (s, d) in enumerate(edges):
                    #     idx_edge = aux_dict.get((s, d), aux_dict.get((s, d)))
                    for idx_edge, (s,d) in enumerate(edges):
                        # key = "{}_edge{}".format(file_name, idx_edge)
                        idx_edge = "{}_{}".format(s, d)
                        key = "{}-edge{}".format(file_name, idx_edge)
                        idx_edge2 = "{}_{}".format(d, s)
                        key2 = "{}-edge{}".format(file_name, idx_edge2)
                        _, hyp_prob = results.get(key, results.get(key2, [0,0]))
                        if hyp_prob >= min_w:
                            edges_pruned.append([s, d])
                    # print(file_name, len(edges), len(edges_pruned))
                    edges = edges_pruned
                    
                    ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                    nodes, edges, labels, edge_features)
                    # print(ids)
                    # exit()
                if len(edges) != len(new_nodes):
                    print("Problem. {} - Edges {} new_nodes {}".format(raw_path, len(edges), len(new_nodes)))
                    exit()
                labels = []
                if self.conjugate == "COL":
                    labels = new_labels_cols
                elif self.conjugate == "ROW":
                    labels = new_labels_rows
                elif self.conjugate == "CELL":
                    labels = new_labels_cells
                elif self.conjugate == "ALL":
                    # labels = zip()
                    if self.onehot:
                        labels = np.zeros((len(new_labels_cells), 4))
                    else:
                        labels = np.zeros(len(new_labels_cells))

                    for idx, _ in enumerate(new_labels_cols):
                        col, row, cell = new_labels_cols[idx], new_labels_rows[idx], new_labels_cells[idx]
                        if row == 1 and col == 1 and cell == 0:
                            print("Error")
                            exit()
                        if cell:
                            # l = [0, 0, 1, 0]
                            if self.onehot:
                                labels[idx, 2] = 1
                            else:
                                labels[idx] = 2
                        elif row:
                            # l = [1, 0, 0, 0]
                            if self.onehot:
                                labels[idx, 0] = 1
                            else:
                                labels[idx] = 0
                        elif col:
                            # l = [0, 1, 0, 0]
                            if self.onehot:
                                labels[idx, 1] = 1
                            else:
                                labels[idx] = 1
                        else:
                            # l = [0, 0, 0, 1] # out of table
                            if self.onehot:
                                labels[idx, 3] = 1
                            else:
                                labels[idx] = 3
                elif self.conjugate == "CR":
                    for idx, _ in enumerate(new_labels_cols):
                        col, row = new_labels_cols[idx], new_labels_rows[idx]
                        labels.append([col, row])

                self.ant_nodes = ant_nodes
                self.ant_edges = ant_edges
                # info_edges = self.add_label_to(new_nodes, labels)
                info_edges = new_edge_feats
                nodes = new_nodes
                edge_index = np.array(new_edges).T
            if len(nodes) != len(labels):
                print("Problem with nodes and labels")
                exit()
            # if edge_index.shape[1] != len(info_edges):
            #     print(edge_index.shape[1])
            #     print(edge_index[:5])
            #     print(len(info_edges))
            #     print(info_edges[:5])
            #     print("Problem with edges")
            #     exit()

            self.ids.extend(ids)
            positions = self.get_positions(nodes)
            # for n in nodes:
            #     print("x.shape ", n.shape)
            # print(type(nodes))
            # nodes = np.array([np.array(a) for a in nodes])
            # print(nodes)
            try:
                x = torch.tensor(nodes, dtype=torch.float)
            except:
                nodes = [x[:-1] for x in nodes]
                x = torch.tensor(nodes, dtype=torch.float)


            node_num = np.array(range(len(labels) + node_numeration))
            node_numeration += len(labels)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            info_edges = torch.tensor(info_edges, dtype=torch.float)
            positions = torch.tensor(positions, dtype=torch.float)
            node_num = torch.tensor(node_num, dtype=torch.long)


            data = Data(x=x,
                        edge_index=edge_index,
                        y=labels_tensor,
                        edge_attr=info_edges,
                        pos=positions,
                        node_num=node_num,
                        )
            # d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            # self.deg += torch.bincount(d, minlength=self.deg.numel())

            self.labels.extend(labels)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            self.flist_processed.append(os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            i += 1
        self.prob_class = self.get_prob_class()
        # print("Deg: ", self.deg)

    def get_positions(self, nodes):
        res = []
        for node in nodes:
            res.append([node[6], node[7]]) # mid point
        return res

    def get(self, idx):
        """Implements the logic to load a single graph.
        Internally, torch_geometric.data.Dataset.__getitem__() gets data objects
        from torch_geometric.data.Dataset.get() and optionally transforms them according to transform.
        """
        data_save = torch.load(osp.join(self.processed_dir_, 'data_{}.pt'.format(idx)))
        if self.transform:
            data_save = self.transform(data_save)
        return data_save


def get_all(path, ext="pkl"):
    if path[-1] != "/":
        path = path+"/"
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == "__main__":
    def prepare():
        """
        Logging and arguments
        :return:
        """

        # Logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        # --- keep this logger at DEBUG level, until aguments are processed
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # --- Get Input Arguments
        in_args = arguments(logger)
        opts = in_args.parse()


        fh = logging.FileHandler(opts.log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # --- restore ch logger to INFO
        ch.setLevel(logging.INFO)

        return logger, opts

    logger, opts = prepare()
    opts.tr_data = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_0.15_lentext"
    opts.fix_class_imbalance = True
    opts.conjugate = "ALL"
    opts.classify = "NO"
    flist = get_all(opts.tr_data)
    dataset_tr = ABPDataset_BIESO(root=opts.tr_data, split="dev", flist=get_all(opts.tr_data), transform=None, opts=opts)
    dataloader = DataLoader(
        dataset_tr,
        batch_size=opts.batch_size,
        shuffle=False,
        # num_workers=opts.num_workers,
        # pin_memory=opts.pin_memory,
    )
    total_nodes = 0
    total_edges = 0
    total_nodes_conjugation = 0
    total_edges_conjugation = 0
    debug = False
    # for raw_path in flist:
    #     # if "0030_S_Grafenau_007-05_0097" not in raw_path:
    #     #     continue
    #
    #     # Read data from `raw_path`.
    #     f = open(raw_path, "rb")
    #     data_load = pickle.load(f)
    #     f.close()
    #     if not debug:
    #         ids = data_load['ids']
    #         nodes = data_load['nodes']
    #         edges = data_load['edges']
    #         labels = data_load['labels']
    #         edge_features = data_load['edge_features']
    #     else:
    #         ids = ["0_0", "1_1", "2_2"]
    #         nodes = [[70,71], [80,81], [90,91]]
    #         edges = [[0,1],[1,0], [0,2],[2,0], [2,1],[1,2] ]
    #         labels = [
    #             {
    #                 'row':0, "col":0,
    #             },
    #             {
    #                 'row': 0, "col": 0,
    #             },
    #             {
    #                 'row': 1, "col": 1,
    #             }
    #         ]
    #         edge_features = [
    #             [0,8,7,6, 0],
    #             [0,8,7,6, 1],
    #             [1,11,12,13, 2],
    #             [1,11,12,13, 3],
    #             [1,14,15,16, 4],
    #             [1,14,15,16, 5],
    #                          ]
    #         print(ids)
    #         print(nodes)
    #         print(edges)
    #         print(labels)
    #         print(edge_features)
    #     total_nodes += len(nodes)
    #     total_edges += len(edges)
    #
    #     ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = dataset_tr.conjugate_nx(ids,
    #         nodes, edges, labels, edge_features)
    #     nodes = new_nodes
    #     edge_index = np.array(new_edges).T
    #     info_edges = new_edge_feats
    #     labels = new_labels_cols
    #     total_nodes_conjugation += len(new_nodes)
    #     total_edges_conjugation += len(new_edges)
    #     # for i in range(len(ids)):
    #     #     id_line = ids[i]
    #     #     label = labels[i]
    #     #     print("{} {}\n".format(id_line, label))
    #     # print(len(nodes))
    #     # print(len(ids))
    #     if debug:
    #         print(ids)
    #         print(nodes)
    #         print(new_edges)
    #         print(new_labels_cols)
    #         print(new_edge_feats)
    dataset_tr.process()
    dataset_tr.get_prob_class()
    res = []
    res_gt = []
    for v_batch, v_sample in enumerate(dataloader):
        # f_names = v_sample.fname
        y_gt = tensor_to_numpy(v_sample.y)
        print(y_gt)
        res_gt.append(y_gt)

    labels = np.hstack(res_gt)
    results_test = list(zip(dataset_tr.ids, labels))
    for i in results_test:
        print(i)
    # print("A total of {} nodes and {} edges".format(total_nodes, total_edges))
    # print("A total of {} nodes and {} edges  in conjugation".format(total_nodes_conjugation, total_edges_conjugation))
    # print("Prob class weight: {}".format(dataset_tr.prob_class))
    # print("A total of {} num nodes to classify".format(len(dataset_tr.labels)))

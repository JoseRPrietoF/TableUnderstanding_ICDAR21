import torch.nn.functional as F
import torch_geometric.nn as geo_nn
import torch
from torch import nn
import torch_geometric as tg
# from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import to_undirected
from torch_geometric.utils import degree
try:
    from models.nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv
except:
    from nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv

class myNNConv(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(myNNConv, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.NNConv = geo_nn.EdgeConv(mlp)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dynamic:
            print("dynamic, ", self.k)
            edge_index = geo_nn.knn_graph(x, self.k, 
            # batch=batch, 
            loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        mlp = Seq(Linear((2 * in_c) + num_edge_features, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        print(mlp)
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvMultNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvMultNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        # num_node_features = dataset.num_node_features
        mlp_nodes = Seq(Linear(2 * in_c, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        mlp_edges = nn.Sequential(
            nn.Linear(num_edge_features + in_c, out_c),
            nn.ReLU(),
            # nn.Linear(num_node_features,num_node_features)
        )

        print(mlp_nodes)
        print(mlp_edges)
        self.NNConv = EdgeFeatsConvMult(
            in_c=in_c,
            out_c=out_c,
            nn_nodes=mlp_nodes,
            nn_edges=mlp_edges,
            root_weight=root_weight
        )

        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class GatConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(GatConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features

        self.NNConv = tg.nn.GATConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index)
        return x, edge_index, edge_attr

class ECNConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(ECNConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = ECNConv(in_channels =in_c,
                                    out_channels =out_c,
                                    num_edge_features=num_edge_features,
                                    # concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index, edge_attr)
        return x, edge_index, edge_attr

class TransformerNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(TransformerNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = geo_nn.TransformerConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    edge_dim = num_edge_features,
                                    concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index, edge_attr)
        return x, edge_index, edge_attr

class Net(torch.nn.Module):

    def __init__(self, dataset, opts):
        """
        model default: EdgeConv
        """
        super(Net, self).__init__()
        self.opts = opts
        layers = opts.layers
        model = opts.model
        model = model.lower()
        # layer_used = "edgeconv"
        if model == "edgeconv":
            layer_used = myNNConv
        elif model == "edgefeatsconv":
            layer_used = EdgeFeatsConvNN
        elif model == "edgefeatsconvmult":
            layer_used = EdgeFeatsConvMultNN
        elif model == "gat":
            layer_used = GatConvNN
        elif model == "ecn":
            layer_used = ECNConvNN
        elif model == "transformer":
            layer_used = TransformerNN
        else:
            print("Model {} not implemented".format(model))
            exit()

        print("Using {} layers".format(layer_used))
        model = [
            EdgeFeatsConvNN(dataset.num_node_features, layers[0], dataset=dataset, opts=opts,),
            layer_used(layers[0], layers[1], dataset=dataset, opts=opts,),
                 ]

        for i in range(2, len(layers)):

            model = model + [
                layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts,),
            ]
        if self.opts.classify != "EDGES":
            if self.opts.g_loss == "NLL":
                model = model + [
                    layer_used(layers[-1], dataset.num_classes, bn=False, dropout=False, dataset=dataset, opts=opts,)
                ]
            else:
                model = model + [
                    layer_used(layers[-1], 1, bn=False, dropout=False, dataset=dataset, opts=opts,)
                ]
        else:
            #+ dataset.num_edge_features
            self.mlp_edges = nn.Sequential(nn.Linear(layers[-1] + dataset.num_edge_features , layers[-1]), nn.ReLU(True), nn.Linear(layers[-1], 2))
        self.model = nn.Sequential(*model)
        self.num_params = 0
        for param in self.parameters():
            self.num_params += param.numel()

    def calc_feats_edges(self, x, edge_index, edge_attr=None):
        s = x[edge_index[0]]
        d = x[edge_index[1]]
        # print(s.size())
        # print(d.size())
        edge_info = (s-d).abs()
        # print(edge_info.size(), edge_attr.size())
        if edge_attr is not None:
            edge_info = torch.cat([edge_info, edge_attr], dim=1)
        # print(x.size())
        feats = self.mlp_edges(edge_info)
        # print(score)
        # exit()

        return feats

    def forward(self, data):
        x, edge_index_orig = data.x, data.edge_index
        edge_attr_orig = data.edge_attr
        x, edge_index, edge_attr = self.model([x, edge_index_orig, edge_attr_orig])
        if self.opts.classify == "EDGES":
            x = self.calc_feats_edges(x, edge_index_orig, edge_attr_orig)
           
        # else 
        if self.opts.g_loss == "NLL":
            return F.log_softmax(x, dim=1)
        else:
            return torch.squeeze(x)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_normal_(m.weight.data)
        except:
            print("object has no attribute 'weight'")
        # init.constant(m.bias.data, 0.0)
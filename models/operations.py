from __future__ import print_function
from __future__ import division
from builtins import range
import torch.nn.functional as F
import torch_geometric.nn as geo_nn
import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.utils import to_undirected
from torch.nn import Sequential as Seq, Linear, ReLU
try:
    from models.nnconv import EdgeFeatsConv, EdgeFeatsConvMult, NodeFeatsConv, NodeFeatsConvv2, NodeFeatsConvv3
except:
    from nnconv import EdgeFeatsConv, EdgeFeatsConvMult, NodeFeatsConv, NodeFeatsConvv2, NodeFeatsConvv3


class ProjectionPooling(nn.Module):
    def __init__(self, axis=2):
        """:param axis: row = 2 | col = 3"""
        super(ProjectionPooling, self).__init__()
        self.axis = axis
        # print("axis: {}".format(self.axis))
    def forward(self, layer):

        mean = layer.mean(self.axis)
        if self.axis == 2:
            for i in range(layer.size()[self.axis]):
                layer[:, :, i, :] = mean
        elif self.axis == 3:
            for i in range(layer.size()[self.axis]):
                layer[:, :, :, i] = mean
        elif self.axis == 1:
            for i in range(layer.size()[self.axis]):
                layer[:, i, :, :] = mean
        return layer


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0.0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_uniform(m.weight.data)
    elif classname.find("BatchNorm2d") != -1:
        # nn.init.xavier_uniform(m.weight.data)
        # init.constant(m.bias.data, 0.0)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def zero_bias(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.constant(m.bias.data, 0.0)


def off_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        m.p = 0.0

def on_dropout(m):
    classname = m.__class__.__name__
    if classname.find("Dropout") != -1:
        m.p = 0.5


class Mish(nn.Module):
    '''
    https://github.com/digantamisra98/Mish/blob/master/Mish/Torch/mish.py
    Applies the mish function element-wise:
    mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Examples:
        >>> m = Mish()
        >>> input = torch.randn(2)
        >>> output = m(input)
    '''
    def __init__(self):
        '''
        Init method.
        '''
        super().__init__()

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        return input * torch.tanh(F.softplus(input))

"""
//////////////////////// NETS //////////////////
"""
class myNNConv(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, **kargs):
        super(myNNConv, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  ReLU(),
                  Linear(out_c, out_c))
        self.NNConv = geo_nn.EdgeConv(mlp)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index_DO, edge_attr_DO  = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index_DO,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, activation=nn.Module, **kargs):
        super(EdgeFeatsConvNN, self).__init__()
        root_weight = opts.root_weight
        self.activation = activation
        # num_edge_features = dataset.num_edge_features
        mlp = Seq(Linear((2 * in_c) + num_edge_features_in, out_c),
                  activation(),
                  Linear(out_c, out_c))
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        # self.NNConv = EdgeFeatsConv(in_c, out_c, mlp)
        # self.NNConv = geo_nn.GraphUNet(in_c, 4, out_c,depth=2, pool_ratios=0.5)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index_DO, edge_attr_DO = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index_DO, edge_attr_DO )
        if self.bn:
            x = self.batchNorm(x)

        x = self.activation()(x)
        return x, edge_index, edge_attr

class NodeFeatsConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, activation=nn.Module, **kargs):
        super(NodeFeatsConvNN, self).__init__()
        self.k = kargs.get("knn", 0)
        self.dynamic = self.k > 0
        root_weight = opts.root_weight
        self.activation = activation
        mlp = Seq(Linear((2 * in_c), out_c),
                  activation(),
                  Linear(out_c, out_c))
        self.NNConv = NodeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data
        if self.dropout:
            if self.dynamic:
                edge_index_DO, _ = tg.utils.dropout_adj(edge_index,
                                                         p=0.1, force_undirected=True,
                                                         )
                x = self.NNConv(x, edge_index_DO, edge_attr)
            else:
                edge_index_DO, edge_attr_DO = tg.utils.dropout_adj(edge_index, edge_attr,
                                                        p=0.1, force_undirected=True,
                                                        )
                x = self.NNConv(x, edge_index_DO, edge_attr_DO )
        else:
            x = self.NNConv(x, edge_index, edge_attr )

        if self.bn:
            x = self.batchNorm(x)
        x = self.activation()(x)
        if self.dynamic:
            edge_index = geo_nn.knn_graph(x, self.k, batch=batch, loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        return x, edge_index, edge_attr, batch

class NodeFeatsConvv2NN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, activation=nn.Module, **kargs):
        super(NodeFeatsConvv2NN, self).__init__()
        root_weight = opts.root_weight
        self.activation = activation
        mlp = Seq(Linear((2 * in_c), out_c),
                  activation(),
                  Linear(out_c, out_c))
        self.NNConv = NodeFeatsConvv2(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr, batch = data
        if self.dropout:
            edge_index_DO, edge_attr_DO = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index_DO, edge_attr_DO )
        if self.bn:
            x = self.batchNorm(x)
        x = self.activation()(x)
        return x, edge_index, edge_attr, batch

class NodeFeatsConvv3NN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, activation=nn.Module, **kargs):
        super(NodeFeatsConvv3NN, self).__init__()
        root_weight = opts.root_weight
        self.activation = activation
        mlp = Seq(Linear(in_c, out_c),
                  activation(),
                  Linear(out_c, out_c))
        self.NNConv = NodeFeatsConvv3(in_c=in_c,
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
            edge_index_DO, edge_attr_DO = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index_DO, edge_attr_DO )
        if self.bn:
            x = self.batchNorm(x)
        x = self.activation()(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvMultNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None, opts={}, activation=ReLU, last=False, **kargs):
        super(EdgeFeatsConvMultNN, self).__init__()
        root_weight = opts.root_weight
        self.activation = activation
        self.last = last
        mlp_nodes = Seq(Linear(2 * in_c, out_c),
                  activation(),
                  Linear(out_c, out_c))
        mlp_edges = nn.Sequential(
            nn.Linear(num_edge_features_in + in_c, out_c),
            activation(),
            # nn.Linear(num_node_features,num_node_features)
        )
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
            edge_index_DO, edge_attr_DO  = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index_DO, edge_attr_DO )
        if not self.last:
            if self.bn:
                x = self.batchNorm(x)
            x = self.activation(x)
        return x, edge_index, edge_attr


class EdgeFeatsConvNN_UpdateEdges(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, num_edge_features_in=None,num_edge_features_out=None, opts={}, last=False, activation=ReLU, **kargs):
        super(EdgeFeatsConvNN_UpdateEdges, self).__init__()
        root_weight = opts.root_weight
        self.activation = activation
        mlp = Seq(Linear((2 * in_c) + num_edge_features_in, out_c),
                  activation(),
                  Linear(out_c, out_c),
                  activation(),)
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )

        self.last = last
        if not last:
            self.mlp_edges_feats = Seq(Linear((out_c*2)+num_edge_features_in, num_edge_features_out ),
                                       activation(),
                                       Linear(num_edge_features_out , num_edge_features_out ),
                                       activation())

        # self.NNConv = EdgeFeatsConv(in_c, out_c, mlp)
        # self.NNConv = geo_nn.GraphUNet(in_c, 4, out_c,depth=2, pool_ratios=0.5)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
        # if self.dropout and self.training: # TODO
            edge_index_DO, edge_attr_DO = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
            x = self.NNConv(x, edge_index_DO, edge_attr_DO)
        else:
            x = self.NNConv(x, edge_index, edge_attr)

        if not self.last:
            if self.bn:
                x = self.batchNorm(x)
            x = self.activation()(x)
            row, col = edge_index
            attrs = torch.cat([x[row], x[col], edge_attr], dim=1)
            edge_attr = self.mlp_edges_feats(attrs)

        return x, edge_index, edge_attr


"""
DENSE
"""

class ActivationBlock(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data
        x = self.activation(x)
        return [x, edge_index, edge_attr, batch]

class Dense_Block(nn.Module):
    def __init__(self, function, activation, first = False, **kwargs):
        super(Dense_Block, self).__init__()
        self.function = function
        self.activation = activation()
        self.first = first

    def forward(self, data):
        x, edge_index, edge_attr, batch = data
        new_x, edge_index, edge_attr, batch = self.function([x, edge_index, edge_attr, batch])
        new_x = self.activation(new_x)
        if not self.first:
            next_x = torch.cat([x, new_x], 1)
        else:
            next_x = new_x
        return [next_x, edge_index, edge_attr, batch]

class Dense_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_features_in, opts, n_blocks = 2,  activation=Mish, function=EdgeFeatsConvNN, **kwargs):
        super(Dense_Net, self).__init__()
        self.activation = activation()
        self.bn = nn.BatchNorm1d(in_channels)
        model = [Dense_Block(function(in_c=in_channels, out_c=out_channels, num_edge_features_in=num_edge_features_in,
                                      activation=activation, opts=opts, **kwargs),
                             activation=activation, first=True,)
                 ]
        # print("Out channels: {}".format(out_channels))

        for i in range(1, n_blocks):
            # print("Out channels: {}".format(out_channels*i))
            model = model + [
                Dense_Block(function(in_c=out_channels*i, out_c=out_channels, num_edge_features_in=num_edge_features_in,
                         activation=activation, opts=opts, **kwargs), activation=activation),
            ]

        self.model = nn.Sequential(*model)

        self.transition = function(in_c=out_channels*n_blocks, out_c=out_channels, num_edge_features_in=num_edge_features_in,  activation=activation, bn=False, opts=opts, **kwargs)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data

        x = self.bn(x)

        c5_dense, edge_index, edge_attr, batch = self.model([x, edge_index, edge_attr, batch])

        last, edge_index, edge_attr, batch = self.transition([c5_dense, edge_index, edge_attr, batch])

        last = self.activation(last)
        return last, edge_index, edge_attr, batch



class Residual_Block(nn.Module):
    def __init__(self, function, activation, first = False):
        super(Residual_Block, self).__init__()
        self.function = function
        self.activation = activation()
        self.first = first

    def forward(self, data):
        x, edge_index, edge_attr = data
        new_x, edge_index, edge_attr = self.function([x, edge_index, edge_attr])
        new_x = self.activation(new_x)
        return [new_x, edge_index, edge_attr]

class Residual_Net(nn.Module):
    def __init__(self, in_channels, out_channels, num_edge_features_in, opts, n_blocks = 3,  activation=Mish, function=EdgeFeatsConvNN, **kargs):
        super(Residual_Net, self).__init__()
        self.activation = activation()
        model1 = [Residual_Block(function(in_c=in_channels, out_c=out_channels, num_edge_features_in=num_edge_features_in,
                                      activation=activation, opts=opts),
                             activation=activation)
                 ]

        model2 = []
        for i in range(1, n_blocks):
            model2 = model2 + [
                Residual_Block(function(in_c=out_channels, out_c=out_channels, num_edge_features_in=num_edge_features_in,
                         activation=activation, opts=opts), activation=activation),
            ]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)


    def forward(self, data):
        x, edge_index, edge_attr = data


        residual, edge_index, edge_attr = self.model1([x, edge_index, edge_attr])
        res, edge_index, edge_attr = self.model2([residual, edge_index, edge_attr])
        res = res + residual

        return res, edge_index, edge_attr


def weighted_binary_cross_entropy(output, target, weights=None):
    output = torch.clamp(output,min=1e-8,max=1-1e-8)
    output = F.sigmoid(output)

    if weights is not None:
        assert len(weights) == 2
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))
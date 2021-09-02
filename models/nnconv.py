import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.inits import reset, uniform
  
import math
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch.nn import Sequential as Seq, Sigmoid, Linear

class NodeFeatsConvv3(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NodeFeatsConvv3, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # cat = torch.cat([x_i, x_j], dim=1)
        return self.nn(x_j)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class NodeFeatsConvv2(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NodeFeatsConvv2, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        cat = torch.cat([x_i, x_j], dim=1)
        return self.nn(cat)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class NodeFeatsConv(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NodeFeatsConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        cat = torch.cat([x_i, x_j - x_i], dim=1)
        return self.nn(cat)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class NodeFeatsConv(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(NodeFeatsConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        cat = torch.cat([x_i, x_j - x_i], dim=1)
        return self.nn(cat)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class EdgeFeatsConv(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(EdgeFeatsConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        cat = torch.cat([x_i, x_j - x_i, pseudo], dim=1)
        return self.nn(cat)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class EdgeFeatsConvMult(MessagePassing):
    def __init__(self,
                 in_c,
                 out_c,
                 nn_nodes,
                 nn_edges,
                 aggr='add', #TODO TEST
                 root_weight = True,
                 bias = True,
                 **kwargs):
        super(EdgeFeatsConvMult, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_c
        self.out_channels = out_c
        self.nn_nodes = nn_nodes
        self.nn_edges = nn_edges
        self.root_weight = root_weight

        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn_nodes)
        reset(self.nn_edges)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)


    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr

        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        cat_edge = torch.cat([x_j, pseudo], dim=1)
        # print(cat_edge.size())
        weight_edges = self.nn_edges(cat_edge)
        # print(weight_edges.size())

        cat = torch.cat([x_i, x_j - x_i], dim=1)
        weight_nodes = self.nn_nodes(cat)
        # print(weight_nodes.size())
        # res = torch.matmul(weight_edges, weight_nodes)
        res = weight_edges * weight_nodes
        # print(res.size())
        # print("-------------------")
        return res

    def __repr__(self):
        if self.root_weight:
            return '{}(nn_nodes={}, nn_edges={}) + root'.format(self.__class__.__name__, self.nn_nodes, self.nn_edges)
        return '{}(nn_nodes={}, nn_edges={})'.format(self.__class__.__name__, self.nn_nodes, self.nn_edges)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

class MultimodalEdgeConv(MessagePassing):
    def __init__(self,
                 size_A,
                 size_B,
                 nn,
                 aggr='max', #TODO TEST
                 root_weight=True,
                 bias=True,
                 **kwargs):
        super(MultimodalEdgeConv, self).__init__(aggr=aggr, **kwargs)
        self.size_A = size_A
        self.size_B = size_B
        self.nn = nn
        self.root_weight = root_weight
        if root_weight:
            self.root = Parameter(torch.Tensor(in_c, out_c))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        uniform(self.in_channels, self.root)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, edge_attr):
        """"""
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, x=x, pseudo=pseudo)

    def message(self, x_i, x_j, pseudo):
        """
        Do the multimodal operations with de edge (pseudo) attributes using size_A and size_B
        """
        
        cat = torch.cat([x_i, x_j - x_i, pseudo], dim=1)
        return self.nn(cat)

    def __repr__(self):
        if self.root_weight:
            return '{}(nn={}) + root'.format(self.__class__.__name__, self.nn)
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)

    def update(self, aggr_out, x):
        if self.root is not None:
            aggr_out = aggr_out + torch.mm(x, self.root)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out



class ECNConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, in_channels: int, out_channels: int,
                 num_edge_features: int,
                 improved: bool = False, cached: bool = False,
                 add_self_loops: bool = True, normalize: bool = True,
                 bias: bool = True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(ECNConv, self).__init__(**kwargs)
        self.edge_mlp = Seq(Linear(num_edge_features, 1),Sigmoid())
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self._cached_edge_index = None
        self._cached_adj_t = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_attr) -> Tensor:
        x = torch.matmul(x, self.weight)
        pseudo = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, pseudo=pseudo,
                             size=None)

        if self.bias is not None:
            out += self.bias

        return out

    #  edge_weight: OptTensor
    def message(self, x_j: Tensor, pseudo) -> Tensor:
        edge_weight = self.edge_mlp(pseudo)
        # return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j
        # print(x_j.size())
        # print(edge_weight.size())
        # exit()
        return x_j * edge_weight

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)                                


def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)
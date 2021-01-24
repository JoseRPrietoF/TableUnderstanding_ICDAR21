import networkx as nx
import numpy as np
import time
"""Functions for generating line graphs."""
from itertools import combinations
from collections import defaultdict
from networkx.utils import arbitrary_element, generate_unique_node
from networkx.utils.decorators import not_implemented_for

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))

        return ret
    return wrap

# @timing
def conjugate_nx(ids, nodes, edges, labels, edge_features, idx=0, list_idx=[]):
    new_labels_cols = []
    new_labels_rows = []
    new_labels_cells = []
    new_ids = []
    G = nx.Graph()
    # print(nodes)
    # print(labels[i])
    # exit()
    # G.add_nodes_from([i for i, j in enumerate(nodes)])
    # G.add_nodes_from([j for i, j in enumerate(nodes)])
    for i, j in enumerate(nodes):
        G.add_node(i, data=j)
    # G.add_nodes_from(nodes)
    # G.add_nodes_from(edges)
    # G.add_edges_from(edges)
    edges_dict = {}
    for num, (i, j) in enumerate(edges):
        G.add_edge(i, j, data=edge_features[num])
        a = "{}_{}".format(i, j)
        edges_dict[a] = num
    
    L = line_graph(G)
    # print(list(L.nodes())[-1])
    # print(nodes[list(L.nodes())[-1][0]])
    # print(nodes[list(L.nodes())[-1][1]])
    # exit()

    new_edges = []
    new_nodes = []
    new_edge_feats = []
    nodes_dict = {}
    # print("-", ids[0])
    fname = ids[0].split(".")[0].split("-edge")[0]
    # print(fname)
    # exit()
    follow_edges = {}
    if len(edges) != len(L.nodes):
        print("Problem. Edges {} new_nodes {}".format(len(edges), len(L.nodes)))
    for num_node, (i, j) in enumerate(L.nodes):
        a = "{}_{}".format(i, j)
        origin = nodes[i]
        target = nodes[j]
        # print(i,j)
        # print(origin)
        # print(target)
        # exit()
        nodes_dict[a] = num_node
        num = edges_dict.get(a, None)
        if num is None:
            a = "{}_{}".format(j, i)
            num = edges_dict.get(a, None)
        if num is None:
            print("Error with {}".format(a))
        follow_edges[num_node] = a
        # print(len(edge_features))
        # print(num)
        feats = np.array(edge_features[num])
        """
            x_coord_mid, y_coord_mid,
            x_node, y_node,  # orig
            x,y, # dest
            prob_edge,
            distance,
            left, top, right, bot, # pos
            o_left, o_top, o_right, o_bot, # overlapping
        """
        # idx = [6, 7, 0, 1]
        # if len(feats) != 4:
        #     print("Problem with feat nodes len")
        #     exit()
        new_nodes.append(feats)

        # Label
        row_i = labels[i]['row']
        col_i = labels[i]['col']

        row_j = labels[j]['row']
        col_j = labels[j]['col']
        # if i in list_idx or j in list_idx:
        #     print("Col from origin {} col from target {} ({} -> {})".format(col_i, col_j, i, j))

        if row_i == row_j and row_i != -1 and row_j != -1:
            new_labels_rows.append(1)
        else:
            new_labels_rows.append(0)

        if col_i == col_j and col_i != -1 and col_j != -1:
            new_labels_cols.append(1)
        else:
            new_labels_cols.append(0)

        if row_i == row_j and row_i != -1 and row_j != -1 and col_i == col_j and col_i != -1 and col_j != -1:
            new_labels_cells.append(1)
        else:
            new_labels_cells.append(0)

        # new_ids.append("{}_edge{}".format(fname, num_node))
        new_ids.append("{}-edge{}".format(fname, a))

    for num, ((i1, j1), (i2, j2)) in enumerate(L.edges):
        if i1 == i2 or i1 == j2:
            pos_node = i1
        else:
            pos_node = j1
        try:
            feat_node = nodes[pos_node]
            # if len(feat_node) != 14:
            #     print("Problem with feat nodes len")
            #     exit()
            new_edge_feats.append(feat_node)
        except:
            print("Error with {}".format(pos_node))
            exit()
        a1 = "{}_{}".format(i1, j1)
        a2 = "{}_{}".format(i2, j2)
        new_edges.append((nodes_dict[a1], nodes_dict[a2]))
    
    return new_ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
           new_edge_feats, list(L.nodes()), list(L.edges())#, follow_edges



def line_graph(G, create_using=None):
    r"""Returns the line graph of the graph or digraph `G`.
    The line graph of a graph `G` has a node for each edge in `G` and an
    edge joining those nodes if the two edges in `G` share a common node. For
    directed graphs, nodes are adjacent exactly when the edges they represent
    form a directed path of length two.
    The nodes of the line graph are 2-tuples of nodes in the original graph (or
    3-tuples for multigraphs, with the key of the edge as the third element).
    For information about self-loops and more discussion, see the **Notes**
    section below.
    Parameters
    ----------
    G : graph
        A NetworkX Graph, DiGraph, MultiGraph, or MultiDigraph.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    Returns
    -------
    L : graph
        The line graph of G.
    Examples
    --------
    >>> import networkx as nx
    >>> G = nx.star_graph(3)
    >>> L = nx.line_graph(G)
    >>> print(sorted(map(sorted, L.edges())))  # makes a 3-clique, K3
    [[(0, 1), (0, 2)], [(0, 1), (0, 3)], [(0, 2), (0, 3)]]
    Notes
    -----
    Graph, node, and edge data are not propagated to the new graph. For
    undirected graphs, the nodes in G must be sortable, otherwise the
    constructed line graph may not be correct.
    *Self-loops in undirected graphs*
    For an undirected graph `G` without multiple edges, each edge can be
    written as a set `\{u, v\}`.  Its line graph `L` has the edges of `G` as
    its nodes. If `x` and `y` are two nodes in `L`, then `\{x, y\}` is an edge
    in `L` if and only if the intersection of `x` and `y` is nonempty. Thus,
    the set of all edges is determined by the set of all pairwise intersections
    of edges in `G`.
    Trivially, every edge in G would have a nonzero intersection with itself,
    and so every node in `L` should have a self-loop. This is not so
    interesting, and the original context of line graphs was with simple
    graphs, which had no self-loops or multiple edges. The line graph was also
    meant to be a simple graph and thus, self-loops in `L` are not part of the
    standard definition of a line graph. In a pairwise intersection matrix,
    this is analogous to excluding the diagonal entries from the line graph
    definition.
    Self-loops and multiple edges in `G` add nodes to `L` in a natural way, and
    do not require any fundamental changes to the definition. It might be
    argued that the self-loops we excluded before should now be included.
    However, the self-loops are still "trivial" in some sense and thus, are
    usually excluded.
    *Self-loops in directed graphs*
    For a directed graph `G` without multiple edges, each edge can be written
    as a tuple `(u, v)`. Its line graph `L` has the edges of `G` as its
    nodes. If `x` and `y` are two nodes in `L`, then `(x, y)` is an edge in `L`
    if and only if the tail of `x` matches the head of `y`, for example, if `x
    = (a, b)` and `y = (b, c)` for some vertices `a`, `b`, and `c` in `G`.
    Due to the directed nature of the edges, it is no longer the case that
    every edge in `G` should have a self-loop in `L`. Now, the only time
    self-loops arise is if a node in `G` itself has a self-loop.  So such
    self-loops are no longer "trivial" but instead, represent essential
    features of the topology of `G`. For this reason, the historical
    development of line digraphs is such that self-loops are included. When the
    graph `G` has multiple edges, once again only superficial changes are
    required to the definition.
    References
    ----------
    * Harary, Frank, and Norman, Robert Z., "Some properties of line digraphs",
      Rend. Circ. Mat. Palermo, II. Ser. 9 (1960), 161--168.
    * Hemminger, R. L.; Beineke, L. W. (1978), "Line graphs and line digraphs",
      in Beineke, L. W.; Wilson, R. J., Selected Topics in Graph Theory,
      Academic Press Inc., pp. 271--305.
    """
    if G.is_directed():
        L = _lg_directed(G, create_using=create_using)
    else:
        L = _lg_undirected(G, selfloops=False, create_using=create_using)
    return L


def _node_func(G):
    """Returns a function which returns a sorted node for line graphs.
    When constructing a line graph for undirected graphs, we must normalize
    the ordering of nodes as they appear in the edge.
    """
    if G.is_multigraph():
        def sorted_node(u, v, key):
            return (u, v, key) if u <= v else (v, u, key)
    else:
        def sorted_node(u, v, d=None):
            return (u, v, d) if u <= v else (v, u, d)
            # return (u, v) if u <= v else (v, u)
    return sorted_node


def _edge_func(G):
    """Returns the edges from G, handling keys for multigraphs as necessary.
    """
    if G.is_multigraph():
        def get_edges(nbunch=None):
            return G.edges(nbunch, keys=True)
    else:
        def get_edges(nbunch=None):
            return G.edges(nbunch, data=True)
    return get_edges


def _sorted_edge(u, v,d=None):
    """Returns a sorted edge.
    During the construction of a line graph for undirected graphs, the data
    structure can be a multigraph even though the line graph will never have
    multiple edges between its nodes.  For this reason, we must make sure not
    to add any edge more than once.  This requires that we build up a list of
    edges to add and then remove all duplicates.  And so, we must normalize
    the representation of the edges.
    """
    # return (u[:2], v[:2]) if u[:2] <= v[:2] else (v[:2], u[:2])
    return (u, v) if u[:2] <= v[:2] else (v, u)


def _lg_directed(G, create_using=None):
    """Returns the line graph L of the (multi)digraph G.
    Edges in G appear as nodes in L, represented as tuples of the form (u,v)
    or (u,v,key) if G is a multidigraph. A node in L corresponding to the edge
    (u,v) is connected to every node corresponding to an edge (v,w).
    Parameters
    ----------
    G : digraph
        A directed graph or directed multigraph.
    create_using : NetworkX graph constructor, optional
       Graph type to create. If graph instance, then cleared before populated.
       Default is to use the same graph class as `G`.
    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Create a graph specific edge function.
    get_edges = _edge_func(G)

    for from_node in get_edges():
        # from_node is: (u,v) or (u,v,key)
        L.add_node(from_node)
        for to_node in get_edges(from_node[1]):
            L.add_edge(from_node, to_node)

    return L


def _lg_undirected(G, selfloops=False, create_using=None):
    """Returns the line graph L of the (multi)graph G.
    Edges in G appear as nodes in L, represented as sorted tuples of the form
    (u,v), or (u,v,key) if G is a multigraph. A node in L corresponding to
    the edge {u,v} is connected to every node corresponding to an edge that
    involves u or v.
    Parameters
    ----------
    G : graph
        An undirected graph or multigraph.
    selfloops : bool
        If `True`, then self-loops are included in the line graph. If `False`,
        they are excluded.
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.
    Notes
    -----
    The standard algorithm for line graphs of undirected graphs does not
    produce self-loops.
    """
    L = nx.empty_graph(0, create_using, default=G.__class__)

    # Graph specific functions for edges and sorted nodes.
    get_edges = _edge_func(G)
    sorted_node = _node_func(G)

    # Determine if we include self-loops or not.
    shift = 0 if selfloops else 1

    # edges = set()
    edges_dict = {}
    for u in G:
        # Label nodes as a sorted tuple of nodes in original graph.
        nodes = [sorted_node(*x) for x in get_edges(u)]
        # print(nodes)

        if len(nodes) == 1:
            # Then the edge will be an isolated node in L.
            L.add_node((nodes[0][0], nodes[0][1]), data=nodes[0][-1])

        # Add a clique of `nodes` to graph. To prevent double adding edges,
        # especially important for multigraphs, we store the edges in
        # canonical form in a set.
        for i, a in enumerate(nodes):
            # edges.update([_sorted_edge(a, b) for b in nodes[i + shift:]])

            for b in nodes[i + shift:]:
                x = _sorted_edge(a, b)
                # print(x)
                edges_dict["{}_{}-{}_{}".format(a[0],a[1],b[0],b[1])] = x

            # [print(a, b) for b in nodes[i + shift:]]
            # [print(a, b) for b in nodes[i + shift:]]
    # exit()
    # edges = list(edges_dict.values())
    edges = []
    for u,v in edges_dict.values():
        edges.append((u[:2],v[:2]))
        # L.add_edge()
    # print(edges)
    # exit()
    L.add_edges_from(edges)

    return L



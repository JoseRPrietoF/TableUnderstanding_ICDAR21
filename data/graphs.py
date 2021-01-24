import networkx as nx
import pickle, os, glob
import matplotlib.pyplot as plt

def get_all( path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path, ext))
    return file_names

path = "/data/READ_ABP_TABLE/dataset111/graphs/"
paths = get_all(path, ext="pkl")
print(paths)
for raw_path in paths:
    print(raw_path)
    # Read data from `raw_path`.
    f = open(raw_path, "rb")
    data = pickle.load(f)
    f.close()

    G = nx.Graph()
    # print(data['nodes'])
    nodes = []
    # nodes = zip(list(range(len(data['nodes']))), data['nodes'])
    for i in range(len(nodes)):
        nodes.append((i, data['nodes'][i]))
    # print(list(nodes))

    G.add_nodes_from(nodes)
    G.add_edges_from(data['edges'])


    # nx.draw_circular(G)
    L = nx.line_graph(G)
    print(sorted(map(sorted, L.edges())))
    print(sorted(map(sorted, L.nodes())))

    break
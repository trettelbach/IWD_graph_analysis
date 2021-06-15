import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from b_extract_trough_transects import read_graph
from datetime import datetime

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def add_params_graph(G, edge_param_dict):
    ''' take entire transect dictionary and
    the original graph G and add the mean/median
    parameter values to the graph edges.

    :param G: trough network graph created
    from skeleton
    :param edge_param_dict: dictionary with
    - key: edge (s, e) and
    - value: list with
        - mean width [m]
        - median width [m]
        - mean depth [m]
        - median depth [m]
        - mean r2
        - median r2
        - ratio of considered transects/trough
        - ratio of water-filled troughs
    :return : graph with added edge_param_dict
    parameters added as edge weights.
    '''
    num_emp = 0
    num_full = 0

    # iterate through all graph edges
    for (s, e) in G.edges():
        # and retrieve information on the corresponding edges from the dictionary
        if (s, e) in edge_param_dict:  # TODO: apparently some (xx) edges aren't in the edge_param_dict. check out why
            G[s][e]['mean_width'] = edge_param_dict[(s, e)][0]
            G[s][e]['median_width'] = edge_param_dict[(s, e)][1]
            G[s][e]['mean_depth'] = edge_param_dict[(s, e)][2]
            G[s][e]['median_depth'] = edge_param_dict[(s, e)][3]
            G[s][e]['mean_r2'] = edge_param_dict[(s, e)][4]
            G[s][e]['median_r2'] = edge_param_dict[(s, e)][5]
            G[s][e]['considered_trans'] = edge_param_dict[(s, e)][6]
            G[s][e]['water_filled'] = edge_param_dict[(s, e)][7]
            num_full += 1
        else:
            print("{} doesn't exist in the edge_param_dict, but only in the Graph.".format(str((s, e))))
            num_emp += 1
    print(num_emp, num_full)


def sink_source_analysis(graph):
    ''' analysze how many sources
    and sinks a graph has

    :param graph: an nx.DiGraph
    :return null: only prints the number of
    sources and sinks respectively
    '''
    sinks = 0
    sources = 0
    degree = graph.degree()
    degree_list = []
    for (n, d) in degree:
        degree_list.append(d)
        if graph.in_degree(n) == 0:
            sources += 1
        elif graph.out_degree(n) == 0:
            sinks += 1

    print("sources: {}".format(sources))
    print("sinks: {}".format(sinks))
    # count = Counter(degree_list)
    # count_sorted = sorted(count, reverse=True)
    # print("subgraph_sizes: {0}".format(count))


def connected_comp_analysis(graph):
    ''' print number of connected components
    and their respective sizes '''
    graph = graph.to_undirected()
    nodes = []
    edges = []
    node_size = []
    edge_size = []
    components = [graph.subgraph(p).copy() for p in nx.connected_components(graph)]
    # print(components, type(components))
    for comp in components:
        nodes.append(comp.nodes())
        edges.append(comp.edges())
    for c in nx.connected_components(graph):
        node_size.append(len(c))
    comp_sizes = Counter(node_size)
    od = OrderedDict(sorted(comp_sizes.items()))
    print(f'number of connected components is: {len(node_size)}')
    print(f'their sizes are: {comp_sizes}')
    for i in edges:
        edge_size.append(len(i))
    print(f'they have {edge_size} edges')

def network_density(graph):
    '''calculate network density of
    graph.

    :param graph: an nx.DiGraph
    :return null: only prints network
    density.
    '''
    # number of existing nodes
    num_nodes = nx.number_of_nodes(graph)
    # number of existing edges
    e_exist = nx.number_of_edges(graph)
    # number of potential edges
    e_pot = 3/2 * (num_nodes+1)
    # density
    dens = e_exist / e_pot
    print(f"num_nodes is: \n\t{num_nodes}")
    print(f"e_exist is: \n\t{e_exist}")
    print(f"e_pot is: \n\t{e_pot}")
    print(f"Absolute network density is: \n\t{dens}")


def betweenness_centrality(graph):
    '''calculate average betweenness centrality
    for all edges.

    :param graph: an nx.DiGraph
    :return null: only prints average
    betweenness centrality.
    '''
    bet_cent = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    print(f"Average betweenness centrality is: \n\t{np.mean(list(bet_cent.values()))}\n "
          f"min: {np.min(list(bet_cent.values()))}; max: {np.max(list(bet_cent.values()))}")


def shortest_path_lengths_connected(graph):
    '''get the shortest path lengths and the average
    shortest path lengths for the largest component of
    the directed graph only + make a histogram.

    :param graph: an nx.DiGraph
    :return null: only prints average shortest path
    length and the network diameter (longest shortest
    path length)
    '''
    # Next, use nx.connected_components to get the list of components,
    components = nx.connected_components(nx.to_undirected(graph))
    # then use the max() command to find the largest one:
    largest_component = max(components, key=len)
    G_largest_sub = nx.subgraph(graph, largest_component)

    short_path_length = []
    for i in nx.shortest_path_length(G_largest_sub, weight='weight'):
        for key, val in i[1].items():
            if val != 0:
                short_path_length.append(val)
    mean = np.mean(short_path_length)
    median = np.median(short_path_length)
    print("The average shortest path length of the largest component is:\n\t{0} m (median: {1} m)".format(mean, median))
    print("The diameter of the graph is:\n\t{} m".format(np.max(short_path_length)))


def shortest_path_lengths_not_connected(graph):
    ''' iterate through list of all connected
    components in a graph to get some analysis
    insights.

    :param graph: an nx.DiGraph
    :return null: only prints average shortest path
    length, the network diameter (longest shortest
    path length), and the number of connected
    components.
    '''
    avg_short_path_length = []
    diameter = []
    for c in nx.connected_components(nx.to_undirected(graph)):
        G_sub = graph.subgraph(c)
        short_path_length = []
        for i in nx.shortest_path_length(G_sub, weight='weight'):
            for key, val in i[1].items():
                if val != 0:
                    short_path_length.append(val)
        avg_short_path_length.append(np.mean(short_path_length))
        diameter.append(np.max(short_path_length))
    print(f"Average shortest path lengths per component (median={np.median(avg_short_path_length)}):\n\t{sorted(avg_short_path_length, reverse=True)} m")
    print(f"Diameter of each connected component (median={np.median(diameter)}):\n\t{sorted(diameter, reverse=True)} m")
    print("Number of connected components in the graph:\n\t{}".format(len(avg_short_path_length)))


def get_total_channel_length(graph):
    ''' calculate the total length of
     all troughs within the study area.

    :param graph:
    :return: : an nx.Graph / nx/DiGraph with true
    length of the edge as weight 'weight'.
    :return null: only prints total length of
    all channels combined.
    '''
    total_length = 0
    for (s, e) in graph.edges:
        total_length += graph[s][e]['weight']
    print("The total length of all channels in the network of the study area is:\n\t{} m".format(round(total_length, 2)))


def do_analysis(graph):
    # general info on number of edges and nodes
    print(nx.info(graph))
    # get sinks and sources
    sink_source_analysis(graph)
    # number of connected components
    connected_comp_analysis(graph)
    # # average shortest path lengths
    # # for all:
    # shortest_path_lengths_not_connected(graph)
    # # for largest component only:
    # shortest_path_lengths_connected(graph)
    # # betweenness centrality
    # betweenness_centrality(graph)
    # # network density
    # network_density(graph)
    # # length of all channels in the network
    # get_total_channel_length(graph)
    print("_______________________")


if __name__ == '__main__':
    startTime = datetime.now()

    # read in 2009 data
    G_09, coord_dict_09 = read_graph(edgelist_loc='./data/a_2009/arf_graph_2009.edgelist',
                                     coord_dict_loc='./data/a_2009/arf_graph_2009_node-coords.npy')

    transect_dict_fitted_2009 = load_obj('./data/a_2009/arf_transect_dict_avg_2009')

    add_params_graph(G_09, transect_dict_fitted_2009)

    # read in 2019 data
    G_19, coord_dict_19 = read_graph(edgelist_loc='./data/b_2019/arf_graph_2019.edgelist',
                                     coord_dict_loc='./data/b_2019/arf_graph_2019_node-coords.npy')

    transect_dict_fitted_2019 = load_obj('./data/b_2019/arf_transect_dict_avg_2019')

    add_params_graph(G_19, transect_dict_fitted_2019)

    # graph analysis 2009
    do_analysis(G_09)
    # graph analysis 2019
    do_analysis(G_19)

    print(datetime.now() - startTime)
    plt.show()

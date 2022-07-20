#!/usr/bin/env python

import sys
import pickle
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter, OrderedDict
from b_extract_trough_transects import read_graph
from datetime import datetime

def load_obj(name):
    with open(name + '', 'rb') as f:
        return pickle.load(f)


def add_params_to_graph(G, edge_param_dict):
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
    :return G_upd: graph with added edge_param_dict
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
            # print(G[s][e]['pts'])
            # probably all of the missing ones are those too close to the image border and thus don't have any transects
            num_emp += 1
    print(f'empty edges: {num_emp}, full edges: {num_full}')
    return G


def analyze_sinks_sources(graph):
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

    # print("sources: {}".format(sources))
    # print("sinks: {}".format(sinks))
    # count = Counter(degree_list)
    # count_sorted = sorted(count, reverse=True)
    # print("subgraph_sizes: {0}".format(count))
    return sinks, sources


def analyze_connected_comp(graph):
    ''' print number of connected components
    and their respective sizes '''
    graph = graph.to_undirected()
    nodes = []
    edges = []
    node_size = []
    edge_sizes = []
    components = [graph.subgraph(p).copy() for p in nx.connected_components(graph)]
    # print(components, type(components))
    for comp in components:
        nodes.append(comp.nodes())
        edges.append(comp.edges())
    for c in nx.connected_components(graph):
        node_size.append(len(c))
    comp_sizes = Counter(node_size)
    od = OrderedDict(sorted(comp_sizes.items()))
    # print(f'number of connected components is: {len(node_size)}')
    # print(f'their sizes are: {comp_sizes}')
    for i in edges:
        edge_sizes.append(len(i))
    # print(f'they have {Counter(edge_sizes)} edges')
    return Counter(edge_sizes)


def get_network_density(graph):
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
    # print(f"num_nodes is: \n\t{num_nodes}")
    # print(f"e_exist is: \n\t{e_exist}")
    # print(f"e_pot is: \n\t{e_pot}")
    # print(f"Absolute network density is: \n\t{dens}")
    return e_pot, dens


def get_betweenness_centrality(graph):
    '''calculate average betweenness centrality
    for all edges.

    :param graph: an nx.DiGraph
    :return null: only prints average
    betweenness centrality.
    '''
    bet_cent = nx.betweenness_centrality(graph, normalized=True, weight='weight')
    print(f"Average betweenness centrality is: \n\t{np.mean(list(bet_cent.values()))}\n "
          f"min: {np.min(list(bet_cent.values()))}; max: {np.max(list(bet_cent.values()))}")


def get_shortest_path_lengths_connected(graph):
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


def get_shortest_path_lengths_not_connected(graph):
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
    # print("The total length of all channels in the network of the study area is:\n\t{} m".format(round(total_length, 2)))
    return round(total_length, 2)


def do_analysis(graph):
    # get sinks and sources
    sinks, sources = analyze_sinks_sources(graph)
    # number of connected components
    connected_comp = analyze_connected_comp(graph)
    # network density
    e_pot, dens = get_network_density(graph)
    # length of all channels in the network
    total_channel_length = get_total_channel_length(graph)
    # print(f'num edges: {graph.number_of_edges()}, num nodes: {graph.number_of_nodes()}')
    return graph.number_of_edges(), graph.number_of_nodes(), connected_comp, sinks, sources, e_pot, dens, total_channel_length


if __name__ == '__main__':
    startTime = datetime.now()

    # edgelist = sys.argv[1]
    # npy = sys.argv[2]
    # dict_avg = sys.argv[3]

    edgelist = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_graph_2009.edgelist'
    npy = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_graph_2009_node-coords.npy'
    dict_avg = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_transect_dict_avg_2009.pkl'

    # read in 2009 data
    G_09, coord_dict_09 = read_graph(edgelist_loc=edgelist,
                                     coord_dict_loc=npy)

    transect_dict_fitted_2009 = load_obj(dict_avg)

    G_upd = add_params_to_graph(G_09, transect_dict_fitted_2009)
    nx.write_edgelist(G_upd, 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_graph_2009_avg_weights.edgelist', data=True, delimiter=';')
                      # data=(('pts', list), ('weight', int), ('mean_width', float), ('median_width', float),
                      #             ('mean_depth', float), ('median_depth', float), ('mean_r2', float), ('median_r2', float),
                      #             ('considered_trans', int), ('water_filled', bool)))

    number_of_edges, number_of_nodes, connected_comp, sinks, sources, e_pot, dens, total_channel_length = do_analysis(G_09)

    with open('E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graph_info/graph_2009.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['name', 'number_of_edges', 'number_of_nodes', 'connected_comp', 'sinks', 'sources', 'voronoi_edges', 'graph_density', 'total_channel_length_m'])
        wr.writerow(['arf_2009', number_of_edges, number_of_nodes, connected_comp, sinks, sources, e_pot, dens, total_channel_length])

    print(datetime.now() - startTime)
    plt.show()

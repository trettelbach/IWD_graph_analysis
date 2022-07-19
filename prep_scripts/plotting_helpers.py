import pickle
import scipy
import scipy.ndimage
import numpy as np
import networkx as nx
import csv
import matplotlib.pyplot as plt
from PIL import Image
from b_extract_trough_transects import read_graph
from a_dem_to_graph import read_data
from datetime import datetime
import matplotlib
# import seaborn as sns
# import pandas as pd
from collections import Counter


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
            # print("{} was empty... (border case maybe?)".format(str((s, e))))
            num_emp += 1
    print(num_emp, num_full)


def plot_graph_by_weight(graph, col_parameter, coord_dict, bg):
    micr = bg

    G_no_borders = graph.copy()
    for (s, e) in graph.edges():
        # print(graph[s][e])
        if col_parameter in graph[s][e]:
            # print(G_no_borders[s][e]['mean_depth'])
            continue
        else:
            G_no_borders.remove_edge(s, e)

    f = plt.figure(1, figsize=(3.75, 2.45), dpi=300)  # 900
    ax = f.add_subplot(1, 1, 1)

    colors = [G_no_borders[s][e][col_parameter] for s, e in G_no_borders.edges()]

    colors_nonnan = []
    for i in colors:
        if i > 0:
            colors_nonnan.append(i)

    colors = np.nan_to_num(colors, nan=np.mean(colors_nonnan), copy=True)

    if col_parameter == 'mean_width':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        # draw edge by weight
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors, edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0, edge_vmax=10)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        # nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=10))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('width [m]')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == 'mean_depth':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors, edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0, edge_vmax=0.5)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        # plt.clim(np.min(colors), np.max(colors))
        # plt.colorbar(edges)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=0.5))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('depth [m]')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == 'mean_r2':
        ax.imshow(micr, cmap='gray', alpha=0)
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color=colors,
                                       edge_cmap=plt.cm.viridis,
                                       width=2, ax=ax, edge_vmin=0.8, edge_vmax=1)
        # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.5, node_color='black')
        # plt.clim(np.min(colors), np.max(colors))
        # plt.colorbar(edges)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.8, vmax=1))
        # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_ticks([0.80, 0.85, 0.90, 0.95, 1.00])
        cbar.set_ticklabels(['0.80', '0.85', '0.90', '0.95', '1.00'])
        cbar.set_label('r2')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
        # plt.title(col_parameter)
        # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\12_all_steps_with_manual_delin\images'
        #             r'graph_9m_2009_no-bg' + col_parameter + '.png', dpi=300)  # , bbox_inches='tight'
        # plt.show()
    elif col_parameter == "centrality":
        # micr = np.fliplr(micr)
        ax.imshow(micr, cmap='gray', alpha=0)
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        # draw directionality
        # (and plot betweenness centrality of edges via color)
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=matplotlib.colors.LogNorm(
                                       vmin=3.285928340474751e-07,
                                       vmax=0.0025840540469493443))  # this one's static
                                   # norm=matplotlib.colors.LogNorm(vmin=np.min(list(nx.edge_betweenness_centrality(graph).values())),
                                   #                    vmax=np.max(list(nx.edge_betweenness_centrality(graph).values()))))  # this one's dynamic
        nx.draw(graph, pos=tmp_coord_dict, arrowstyle='->', arrowsize=3.5, width=1, with_labels=False, node_size=0.005,
                edge_color=np.log(np.array(list(nx.edge_betweenness_centrality(graph).values()))), node_color='black',
                edge_cmap=cmap)
        print(np.min(np.array(list(nx.edge_betweenness_centrality(graph).values()))))
        print(np.max(np.array(list(nx.edge_betweenness_centrality(graph).values()))))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('betweenness centrality')
        plt.margins(0.0)
        plt.axis('off')
        plt.tight_layout()
    elif col_parameter == 'directionality':
        # draw directionality
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]

        # without bg
        ax.imshow(micr, cmap='Greys_r', alpha=1)
        nx.draw(graph, pos=tmp_coord_dict, arrowstyle='-', arrowsize=3.5, width=0.8, with_labels=False, node_size=0,
                node_color='seagreen', edge_color='black')
        # with bg
        # ax.imshow(micr, cmap='Greens', alpha=0)
        # nx.draw(graph, pos=tmp_coord_dict, arrowstyle='->', arrowsize=3.5, width=0.8, with_labels=False, node_size=0.65,
        #         node_color='white', edge_color='black')
        # plt.title("directionality")
        plt.margins(0.0)
        plt.axis('off')
        plt.tight_layout()
    elif col_parameter == 'water_filled':
        tmp_coord_dict = {}
        tmp_keys = []
        tmp_vals = []
        for key, val in coord_dict.items():
            tmp_keys.append(key)
            tmp_vals.append(val)
        for i in range(len(tmp_keys)):
            tmp_coord_dict[tmp_keys[i]] = [tmp_vals[i][1], tmp_vals[i][0]]
        ax.imshow(micr, cmap='gray', alpha=0)
        # draw edge by weight
        edges = nx.draw_networkx_edges(G_no_borders, pos=tmp_coord_dict, arrows=False, edge_color="blue", edge_cmap=plt.cm.viridis,
                                       width=colors*2, ax=ax, edge_vmin=0, edge_vmax=1)
                                       # , edge_vmin=np.min(colors), edge_vmax=np.max(colors)
        # # edges = nx.draw_networkx_edges(G_no_borders, pos=coord_dict, arrows=False, edge_color='blue', width=0.75, ax=ax)
        # nodes = nx.draw_networkx_nodes(G_no_borders, pos=tmp_coord_dict, node_size=0.05, node_color='blue')
        cmap = plt.cm.viridis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
                                              # norm=plt.Normalize(vmin=np.min(colors), vmax=np.max(colors))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('fraction')
        # plt.gca().invert_xaxis()
        plt.axis('off')
        plt.margins(0.0)
        plt.tight_layout()
    else:
        print("please choose one of the following col_parameters: 'mean_width', 'mean_depth', 'mean_r2', 'centrality', 'directionality'")


def scatter_depth_width(edge_param_dict, type):
    # print(edge_param_dict)
    width_mean = []
    width_med = []
    depth_mean = []
    depth_med = []
    r2_mean = []
    r2_med = []
    water = []
    for params in edge_param_dict.values():
        if not np.isnan(params[0]):
            width_mean.append(params[0])
            width_med.append(params[1])
            depth_mean.append(params[2])
            depth_med.append(params[3])
            r2_mean.append(params[4])
            r2_med.append(params[5])
            water.append(params[7])
            if params[7] != 0:
                print(params)

    if type == 'median':
        width = width_med
        depth = depth_med
        r2 = r2_med
    elif type == 'mean':
        width = width_mean
        depth = depth_mean
        r2 = r2_mean
    else:
        print("please provide either the keyword 'mean' or 'median'.")

    plt.figure()
    fig = plt.scatter(width, depth, s=3, c=water, cmap='viridis_r')
    plt.xlabel('Width [m]')
    plt.ylabel('Depth [m]')
    plt.title('Trough %s parameters by water presence' % type)
    plt.colorbar(fig)

    # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\11_trough_analysis\scatter_depth_width_r2_%s_orig' % type, dpi=600)


def density_depth_width(edge_param_dict, type):
    """
    makes a density plot.
    has nothing to do with graph density
    :param edge_param_dict:
    :param type:
    :return:
    """
    width_mean = []
    width_med = []
    depth_mean = []
    depth_med = []
    r2_mean = []
    r2_med = []
    for params in edge_param_dict.values():
        if not np.isnan(params[0]):
            # print(params)
            width_mean.append(params[0])
            width_med.append(params[1])
            depth_mean.append(params[2])
            depth_med.append(params[3])
            r2_mean.append(params[4])
            r2_med.append(params[5])

    if type == 'median':
        width = width_med
        depth = depth_med
        r2 = r2_med
    elif type == 'mean':
        width = width_mean
        depth = depth_mean
        r2 = r2_mean
    else:
        print("please provide either the keyword 'mean' or 'median'.")

    plt.figure()
    plt.hist2d(width, depth, (50, 50), cmap=plt.cm.viridis)
    plt.colorbar()
    plt.xlabel('Width [m]')
    plt.ylabel('Depth [m]')
    plt.title('Trough %s parameter density' % type)
    width_sorted = np.sort(width)
    depth_sorted = np.sort(depth)

    print(np.median(width_sorted[-960:]), np.median(depth_sorted[-960:]))
    # plt.savefig(r'D:\01_anaktuvuk_river_fire\00_working\01_processed-data\11_trough_analysis\density_depth_width_%s_detr' % type, dpi=600)

#
# def pairplots(edge_param_dict_09, edge_param_dict_19):
#     width_mean_09 = []
#     width_med_09 = []
#     depth_mean_09 = []
#     depth_med_09 = []
#     r2_mean_09 = []
#     r2_med_09 = []
#     used_09 = []
#     water_09 = []
#     for params in edge_param_dict_09.values():
#         if not np.isnan(params[0]):
#             # print(params)
#             width_mean_09.append(params[0])
#             width_med_09.append(params[1])
#             depth_mean_09.append(params[2])
#             depth_med_09.append(params[3])
#             r2_mean_09.append(params[4])
#             r2_med_09.append(params[5])
#             used_09.append(params[6])
#             water_09.append(params[7])
#     df_09 = pd.DataFrame(zip(width_mean_09, depth_mean_09, r2_mean_09, water_09))
#     df_09.columns = ['width', 'depth', 'r2', 'water']
#
#
#     width_mean_19 = []
#     width_med_19 = []
#     depth_mean_19 = []
#     depth_med_19 = []
#     r2_mean_19 = []
#     r2_med_19 = []
#     used_19 = []
#     water_19 = []
#     for params in edge_param_dict_19.values():
#         if not np.isnan(params[0]):
#             # print(params)
#             width_mean_19.append(params[0])
#             width_med_19.append(params[1])
#             depth_mean_19.append(params[2])
#             depth_med_19.append(params[3])
#             r2_mean_19.append(params[4])
#             r2_med_19.append(params[5])
#             used_19.append(params[6])
#             water_19.append(params[7])
#     df_19 = pd.DataFrame(zip(width_mean_19, depth_mean_19, r2_mean_19, water_19))
#     df_19.columns = ['width', 'depth', 'r2', 'water']
#
#     df = pd.concat([df_19, df_09], keys=['2019', '2009']).reset_index()
#     # print(df)
#     g = sns.pairplot(df, vars=['width', 'depth', 'r2', 'water'], hue='level_0', markers=['o', '+'],
#                      palette=['salmon', 'teal'],
#                      diag_kind="hist",
#                      plot_kws={"alpha": 0.5, 's': 25, 'lw': 0.5},
#                      diag_kws={'alpha': 1, 'bins': 20, 'histtype': 'step'})
#
#     g.axes[2, 1].set_xlim((0.775, 1.025))
#     g.axes[3, 1].set_xlim((-0.05, 1.05))
#     g.axes[1, 2].set_xlim((0.775, 1.025))
#     g.axes[1, 3].set_xlim((-0.05, 1.05))
#     # g.map_lower(sns.kdeplot, levels=6, color=".2")
#     g._legend.set_bbox_to_anchor((0.5, 0.5))
#


def plot_transect_locations(transect_dict, dem):
    emp = np.zeros(dem.shape)
    print(emp.shape)
    # i = 0
    print(type(transect_dict))
    for outer_keys, transects in transect_dict.items():
        # print(outer_keys)
        # if i == 15 or i == 25 or i == 150:
        for tr_coord, infos in transects.items():
            # print(tr_coord, infos)
            if len(infos[0]) > 0:
                # color the skeleton by fit (r2)
                for i in infos[1]:
                    if i[0] < 300 and i[1] < 300:
                        emp[i[0], i[1]] = 2

                emp[tr_coord[0], tr_coord[1]] = 1
        # break
        # i += 1

    plt.figure()
    plt.imshow(emp)
    plt.colorbar()
    plt.clim(vmax=2, vmin=0)
    plt.title("plot the transects ALL")

    plt.savefig('D:/01_anaktuvuk_river_fire/00_working/01_processed-data/12_all_steps_with_manual_delin/images/'
                'all_transects.png', dpi=300)


def plot_connected_components_size(graph):
    '''make histogram of number of nodes in each
    connected component of entire, disconnected graph.'''
    G_u = nx.to_undirected(graph)
    subgraph_sizes = []
    for sub in list(nx.connected_components(G_u)):
        print(type(sub))
        subgraph_sizes.append(len(sub))
    # subgraph_sizes = sorted(subgraph_sizes, reverse=True)
    # print(subgraph_sizes)
    count = Counter(subgraph_sizes)
    # count_sorted = sorted(count, reverse=True)
    print("subgraph_sizes: {0}".format(subgraph_sizes))
    print(count)

    # plt.figure()
    # plt.hist(subgraph_sizes, bins=np.max(subgraph_sizes))
    # plt.title("sub-component sizes")
    # plt.xlabel("number of nodes")
    # plt.ylabel("frequency")


def node_degree_hist(graph1, graph2):
    ''' plot histogram of node degrees
    with degree average.'''
    # first 2009
    degree1 = graph1.degree()
    degree_list1 = []
    for (n, d) in degree1:
        if d != 2:
            degree_list1.append(d)

    av_degree1 = sum(degree_list1) / len(degree_list1)

    print('The average degree for 2009 is {}'.format(np.round(av_degree1, 2)))

    # now 2019
    degree2 = graph2.degree()
    degree_list2 = []
    for (n, d) in degree2:
        if d != 2:
            degree_list2.append(d)

    av_degree2 = sum(degree_list2) / len(degree_list2)
    print(np.max(degree_list1), np.max(degree_list2))
    print('The average degree for 2019 is {}'.format(np.round(av_degree2, 2)))
    # plt.bar(np.array(x1) - 0.15, y1, width=0.3)
    # plt.bar(np.array(x2) + 0.15, y2, width=0.3)
    x_multi = [degree_list1, degree_list2]
    plt.figure(figsize=(4.5, 4.5))
    # plt.style.use('bmh')
    plt.grid(color='gray', linestyle='--', linewidth=0.2, which='both')
    plt.hist(x_multi, bins=range(1, np.max(degree_list2)+2), density=True, rwidth=0.5, color=['salmon', 'teal'],
             label=['2009', '2019'])
    plt.axvline(av_degree1, linestyle='--', color='salmon', linewidth=.9)
    plt.axvline(av_degree2, linestyle='--', color='teal', linewidth=.9)
    plt.xticks([1.5, 3.5, 4.5, 5.5, 6.5], [1, 3, 4, 5, 6])
    plt.text(av_degree1 - 0.175, 0.025, 'mean 2009 = {}'.format(np.round(av_degree1, 2)), rotation=90, fontsize=8)
    plt.text(av_degree2 - 0.175, 0.025, 'mean 2019 = {}'.format(np.round(av_degree2, 2)), rotation=90, fontsize=8)
    plt.text(5.3, 0.015, np.round(x_multi[0].count(5)/len(x_multi[0]), 4), rotation=90, fontsize=8)
    plt.text(5.55, 0.035, np.round(x_multi[1].count(5)/len(x_multi[1]), 4), rotation=90, fontsize=8)
    plt.text(6.3, 0.015, np.round(x_multi[0].count(6)/len(x_multi[0]), 4), rotation=90, fontsize=8)
    plt.text(6.55, 0.015, np.round(x_multi[1].count(6)/len(x_multi[1]), 4), rotation=90, fontsize=8)
    plt.legend(frameon=False)
    plt.ylabel('nodes frequency')
    plt.xlabel('degree')
    plt.savefig('./figures/node_degree_hist.png')

    print(f'{np.round(x_multi[0].count(3)/len(x_multi[0]), 4)} have 3 edges in 2009')
    print(f'{np.round(x_multi[1].count(3)/len(x_multi[1]), 4)} have 3 edges in 2019')
    print(f'{np.round(x_multi[0].count(1)/len(x_multi[0]), 4)} have 1 edge in 2009 -- total: {x_multi[0].count(1)}')
    print(f'{np.round(x_multi[1].count(1)/len(x_multi[1]), 4)} have 1 edge in 2019 -- total: {x_multi[1].count(1)}')


def plot_transect_ontop():
    # key[0], key[1], t, data, data_gauss_fit, val
    # rows = []
    # Cross = ["#c969a1", "#ce4441", "#ee8577", "#eb7926", "#ffbb44", "#859b6c", "#62929a", "#004f63", "#122451", "#122451"]
    # # order=(4, 7, 1, 8, 2, 6, 3, 5, 9), colorblind=False),
    # # Moth = '#4a3a3b', '#984136', '#c26a7a', '#ecc0a1', '#f0f0e4']
    # hiroshige = ["#e76254", "#ef8a47", "#f7aa58", "#ffd06f", "#ffe6b7", "#aadce0", "#72bcd5", "#528fad", "#376795", "#1e466e"]
    # gb = ["#F1BB7B", "#FD6467", "#5B1A18", "#D67236"]
    # hokusai1 = ["#6d2f20", "#b75347", "#df7e66", "#e09351", "#edc775", "#94b594", "#224b5e"]
    hokusai1_sh = ["#b75347", "#e09351", "#94b594", "#224b5e"]
    symbols_dash = ["o:", ">:", "P:", "d:"]
    symbols = ["o-", ">-", "P-", "d-"]

    fig = plt.figure(figsize=(5, 3))
    # fig = pylab.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.yaxis.grid(color='gray', alpha=0.2)
    ax.set_ylabel("depth below surface [m]")
    ax.set_xlabel("transect length [m]")
    ax.set_xticks(list(range(0, 13)))
    ax.axvline(x=6, ls='--', color='gray', alpha=0.5)
    ax.text(-0.2, -0.202, '...', fontsize=10, fontweight='bold')
    ax.text(-0.2, -.218, '__', fontsize=10, fontweight='bold')
    ax.text(0.5, -0.205, 'DTM elevation', fontsize=8)
    ax.text(0.5, -.225, 'Gaussian fit', fontsize=8)
    with open('./figures/transects_to_plot_extra.csv', newline='') as file:
        csvreader = csv.reader(file)
        # row = key[0], key[1], t, data, data_gauss_fit, val
        for (row, i) in zip(csvreader, list(range(0, 10))):
            x = row[0]
            y = row[1]
            t = list(map(float, row[2:15]))
            data = list(map(float, row[15:28]))
            data = list(map(lambda elem: (-1)*elem, data))
            data_gauss_fit = list(map(float, row[28:41]))
            print(data_gauss_fit[0], data_gauss_fit[-1], "---", data[0], data[-1])
            data_gauss_fit = list(map(lambda elem: (-1)*elem, data_gauss_fit))
            ax.plot(t, data, symbols_dash[i], label='DTM elevation', color=hokusai1_sh[i], alpha=0.5)
            ax.plot(t, data_gauss_fit, symbols[i], color=hokusai1_sh[i], alpha=1)
            plt.tight_layout()
            plt.savefig('./figures/fitted_to_coords_4_ex_narrow.png')

    plt.show()


if __name__ == '__main__':
    startTime = datetime.now()

    # read in 2009 data
    # G_09, coord_dict_09 = read_graph(edgelist_loc='./data/a_2009/arf_graph_2009.edgelist',
    #                                  coord_dict_loc='./data/a_2009/arf_graph_2009_node-coords.npy')
    #
    # transect_dict_fitted_2009 = load_obj('./data/a_2009/arf_transect_dict_avg_2009')
    #
    # add_params_graph(G_09, transect_dict_fitted_2009)
    #
    # dem2009 = Image.open('./data/a_2009/arf_dtm_2009.tif')
    # img_det2009 = read_data('./data/a_2009/arf_microtopo_2009.tif')
    # dem2009 = np.array(dem2009)


    # read in 2019 data
    # read in 2019 data
    G_19, coord_dict_19 = read_graph(edgelist_loc='./data/2019/arf_graph_2019.edgelist',
                                     coord_dict_loc='./data/2019/arf_graph_2019_node-coords.npy')

    transect_dict_fitted_2019 = load_obj('./data/2019/arf_transect_dict_avg_2019')

    add_params_graph(G_19, transect_dict_fitted_2019)

    dem2019 = Image.open('./data/2019/arf_dtm_2019.tif')
    img_det2019 = read_data('./data/2019/li-dem_1m_sa3_fill_detrended_16m.tif')
    # dem2019 = np.array(dem2019)
    # plot_transect_locations(transect_dict_fitted_2019, dem2019)
    # node_degree_hist(G_09, G_19)

    # density_depth_width(transect_dict_fitted_2009, 'mean')
    # scatter_depth_width(transect_dict_fitted_2019, 'mean')
    # pairplots(transect_dict_fitted_2009, transect_dict_fitted_2019)

    # for s, e in G_19.edges():
    #     print(G_19[s][e])

    # plot_graph_by_weight(G_19, "mean_width", coord_dict_19, dem2019)
    # plt.savefig('D:/00_orga/04_proposals_presentations/2021-12-13_agu/figures/mean_width_01_wo-bg_2019')
    # plt.show()
    # plot_graph_by_weight(G_19, "mean_depth", coord_dict_19, dem2019)
    # plt.savefig('./figures/mean_depth_01_wo-bg_2019')
    # plt.show()
    # plot_graph_by_weight(G_19, "mean_r2", coord_dict_19, dem2019)
    # plt.savefig('./figures/mean_r2_01_wo-bg_2019')
    # plt.show()
    plot_graph_by_weight(G_19, "directionality", coord_dict_19, img_det2019)
    plt.savefig('D:/00_orga/10_paper_writing/05_ssdbm_graph_analysis/graph_on_detrended_2019.png')
    plt.show()
    # plot_graph_by_weight(G_19, "centrality", coord_dict_19, dem2019)
    # plt.savefig('./figures/centrality_01_wo-bg_2019')
    # plt.show()
    # plot_graph_by_weight(G_19, "water_filled", coord_dict_19, dem2019)
    # plt.savefig('D:/00_orga/04_proposals_presentations/2021-12-13_agu/figures/water_01_wo-bg_2019')
    # plt.show()
    # #
    # plot_graph_by_weight(G_09, "mean_width", coord_dict_09, dem2009)
    # plt.savefig('D:/00_orga/04_proposals_presentations/2021-12-13_agu/figures/mean_width_01_wo-bg_2009')
    # plt.show()
    # plot_graph_by_weight(G_09, "mean_depth", coord_dict_09, dem2009)
    # plt.savefig('./figures/mean_depth_01_wo-bg_2009')
    # plt.show()
    # plot_graph_by_weight(G_09, "mean_r2", coord_dict_09, dem2009)
    # plt.savefig('./figures/mean_r2_01_wo-bg_2009')
    # plt.show()
    # plot_graph_by_weight(G_09, "directionality", coord_dict_09, img_det2009)
    # plt.savefig('D:/00_orga/04_proposals_presentations/2021-12-13_agu/figures/directionality_2009_zoom.png')
    # plt.show()
    # plot_graph_by_weight(G_09, "centrality", coord_dict_09, dem2009)
    # plt.savefig('./figures/centrality_01_wo-bg_2009')
    # plt.show()
    # plot_graph_by_weight(G_19, "water_filled", coord_dict_09, dem2009)
    # plt.savefig('D:/00_orga/04_proposals_presentations/2021-12-13_agu/figures/water_01_wo-bg_2009')
    # plt.show()


    # G_19_empty = G_19.edge_subgraph(border_edges_19).copy()
    # plt.figure()
    # edges2 = nx.draw_networkx_edges(G_19, pos=coord_dict_19, arrows=False, edge_color='blue', width=0.2)
    # edges = nx.draw_networkx_edges(G_19_empty, pos=coord_dict_19, arrows=False, edge_color='red', width=2)
    # nodes = nx.draw_networkx_nodes(G_19_empty, pos=coord_dict_19, node_size=2, node_color='black')


    # G_09_empty = G_09.edge_subgraph(empty_edges_09).copy()
    # plt.figure()
    # edges2 = nx.draw_networkx_edges(G_09, pos=coord_dict_09, arrows=False, edge_color='blue', width=0.2)
    # edges = nx.draw_networkx_edges(G_09_empty, pos=coord_dict_09, arrows=False, edge_color='red', width=2)
    # nodes = nx.draw_networkx_nodes(G_09_empty, pos=coord_dict_09, node_size=2, node_color='black')

    # node_degree_hist(G_09, G_19)

    # plt.figure()
    # plt.imshow(dem2019, cmap='gray', alpha=50)

    # density_depth_width(edge_param_dict=transect_dict_fitted_2009, type="median")
    # density_depth_width(edge_param_dict=transect_dict_fitted_2019, type="median")

    # plot_transect_ontop()


    print(datetime.now() - startTime)
    # plt.show()

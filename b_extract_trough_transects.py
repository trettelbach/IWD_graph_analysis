import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys
import networkx as nx
import pickle

from datetime import datetime
np.set_printoptions(threshold=sys.maxsize)

def read_graph(edgelist_loc, coord_dict_loc):
    ''' load graph and dict containing coords
    of graph nodes

    :param edgelist_loc: path on disk to the
    graph's edgelist
    :param coord_dict_loc: path on disk to
    the coord_dict_loc
    :return G: rebuilt nx.DiGraph from edgelist
    :return: coord_dict: dictionary with node
    coordinates
    '''
    # read edgelist to build graph
    # we don't use 'read_weighted_edgelist' bc we have two weights and we want
    # to gather both. rwe somehow cannot cope with this.
    # the first weight 'weight' actually characterizes the length in pixels of the trough.

    # original dataset
    G = nx.read_edgelist(edgelist_loc, data=True, create_using=nx.DiGraph())
    coord_dict = np.load(coord_dict_loc, allow_pickle=True).item()

    return G, coord_dict


def get_transects(graph, dem, width):
    ''' extract the height from DEM along transects
    perpendicular to the trough line (the graph edge)

    When considering pixels a and b, the transect is
    computed along the perpendicular at px a

    transects work for 4- and 8-connected trough-lines.
    a transect can either be vertical, horizontal, or
    diagonal (the previous are always (width*2+1) pixels
    in width, but in diagonal, these are obv. sqrt(2)
    units longer).

    :param graph: nx.DiGraph (the trough network graph)
    :param dem: np.array of the DEM image
    :param width: int --> how wide should the transect be?
    :return dict_outer: a dictionary with
    - outer_keys: edge (s, e) and
    - outer_values: dict of transects
    with:
    - inner_keys: pixel-coords of trough pixels (x, y)
    inbetween (s, e).
    - inner_values: list with transect coordinates and info:
        - [0]: height information of transect at loc (xi, yi)
        - [1]: pixel coordinates of transect (xi, yi)
            --> len[1] == width*2 + 1
        - [2]: directionality of transect
        - [3]: transect scenario (see publication)
        - [4]: presence of water
    '''
    inner_dictio = []
    edge_val = []
    empty_edge_nodes = []
    for (s, e) in graph.edges():
        values_inner = []
        keys_inner = []
        ps = graph[s][e]['pts']  # pixel coordinates for all trough pixels between edge (s, e)
        # iterate through all pixels of a trough except the first and last one and retrieve transect information
        # this skips troughs with length (2 pixels), but they do not hold much information anyway.
        for i in range(1, len(ps)-1):
            water = False
            keys_inner.append((ps[i][0], ps[i][1]))  # x and y coordinates of each of my current edge pixels
            px_current = (ps[i][0], ps[i][1])  # coords of pixel p
            px_prev = (ps[i-1][0], ps[i-1][1])  # coords of pixel p-1
            px_subs = (ps[i+1][0], ps[i+1][1])  # coords of pixel p+1
            # subset square is needed for diagonal calculation --> subset_dem
            subset_dem = dem[px_current[0] - width:px_current[0] + width + 1, px_current[1] - width:px_current[1] + width + 1]

            # make sure to not consider any cases at the border of the image
            # to avoid only partial transects
            if width < px_current[0] < dem.shape[0]-width and width < px_current[1] < dem.shape[1]-width:
                # now consider the five possible scenarios (see publication for details)

                # if p, p+1 and p-1 are in the same row (so x-value is the same), do:
                # this is scenario a; transect is vertical
                if px_prev[0] == px_current[0] == px_subs[0]:
                    trans_start = px_current[0] - width  # where the transect begins
                    trans_end = px_current[0] + width  # where the transect ends
                    cons = px_current[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start, trans_end+1))
                    y = [cons]*len(x)
                    transect_loc = list(zip(x, y))
                    t_type = "vertical"
                    t_cat = "a"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario b; transect is vertical
                elif px_prev[0] == px_subs[0] and px_prev[0] != px_current[0]:
                    trans_start = px_current[0] - width  # where the transect begins
                    trans_end = px_current[0] + width  # where the transect ends
                    cons = px_current[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start, trans_end+1))
                    y = [cons]*len(x)
                    transect_loc = list(zip(x, y))
                    t_type = "vertical"
                    t_cat = "b"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario e; transect is vertical
                elif px_prev[0] != px_current[0] and px_prev[1] != px_current[1] and px_subs[0] == px_current[0] and px_subs[1] != px_current[1]:
                    trans_start = px_current[0] - width  # where the transect begins
                    trans_end = px_current[0] + width  # where the transect ends
                    cons = px_current[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start, trans_end+1))
                    y = [cons]*len(x)
                    transect_loc = list(zip(x, y))
                    t_type = "vertical"
                    t_cat = "e"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario d; transect is vertical
                elif px_prev[0] == px_current[0] and px_prev[1] != px_current[1] and px_subs[0] != px_current[0] and px_subs[1] != px_current[1]:
                    trans_start = px_current[0] - width  # where the transect begins
                    trans_end = px_current[0] + width  # where the transect ends
                    cons = px_current[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start, trans_end+1))
                    y = [cons]*len(x)
                    transect_loc = list(zip(x, y))
                    t_type = "vertical"
                    t_cat = "d"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario a; transect is horizontal
                elif px_prev[1] == px_current[1] == px_subs[1]:
                    trans_start = px_current[1] - width  # where the transect begins
                    trans_end = px_current[1] + width  # where the transect ends
                    cons = px_current[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start, trans_end+1))
                    x = [cons]*len(y)
                    transect_loc = list(zip(x, y))
                    t_type = "horizontal"
                    t_cat = "a"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario b; transect is horizontal
                elif px_prev[1] == px_subs[1] and px_prev[1] != px_current[1]:
                    trans_start = px_current[1] - width  # where the transect begins
                    trans_end = px_current[1] + width  # where the transect ends
                    cons = px_current[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start, trans_end+1))
                    x = [cons]*len(y)
                    transect_loc = list(zip(x, y))
                    t_type = "horizontal"
                    t_cat = "b"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario e; transect is horizontal
                elif px_prev[0] != px_current[0] and px_prev[1] != px_current[1] and px_subs[1] == px_current[1] and px_subs[0] != px_current[0]:
                    trans_start = px_current[1] - width  # where the transect begins
                    trans_end = px_current[1] + width  # where the transect ends
                    cons = px_current[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start, trans_end+1))
                    x = [cons]*len(y)
                    transect_loc = list(zip(x, y))
                    t_type = "horizontal"
                    t_cat = "e"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario d; transect is horizontal
                elif px_prev[1] == px_current[1] and px_prev[0] != px_current[0] and px_subs[0] != px_current[0] and px_subs[1] != px_current[1]:
                    trans_start = px_current[1] - width  # where the transect begins
                    trans_end = px_current[1] + width  # where the transect ends
                    cons = px_current[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dem[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start, trans_end+1))
                    x = [cons]*len(y)
                    transect_loc = list(zip(x, y))
                    t_type = "horizontal"
                    t_cat = "d"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] > px_current[0] > px_subs[0] and px_prev[1] < px_current[1] < px_subs[1]:
                    transect_heights = subset_dem.diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] - width,  px_current[1] + width + 1))
                    transect_loc = list(zip(x, y))
                    t_type = "diagonal"  # transect is ul to lr
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] < px_current[0] < px_subs[0] and px_prev[1] > px_current[1] > px_subs[1]:
                    transect_heights = subset_dem.diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] - width,  px_current[1] + width + 1))
                    transect_loc = list(zip(x, y))
                    t_type = "diagonal"  # transect is ul to lr
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] < px_current[0] < px_subs[0] and px_prev[1] < px_current[1] < px_subs[1]:
                    transect_heights = np.fliplr(subset_dem).diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] + width,  px_current[1] - (width + 1), -1))
                    transect_loc = list(zip(x, y))
                    t_type = "diagonal"  # transect is ll to ur
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] > px_current[0] > px_subs[0] and px_prev[1] > px_current[1] > px_subs[1]:
                    transect_heights = np.fliplr(subset_dem).diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] + width,  px_current[1] - (width + 1), -1))
                    transect_loc = list(zip(x, y))
                    t_type = "diagonal"  # transect is ll to ur
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # for catching errors...
                else:
                    print("I messed up an edge case...")
                    print("px_prev = {0}, px_current = {1}, px_subs = {2}".format(px_prev, px_current, px_subs))
            else:
                pass
                # print(s, e)  # these are the border cases, but they still have some transects, so all good
        # now recombine all elements to the inner transect dict
        dict_inner = dict(zip(keys_inner, values_inner))  # values_inner ist auch schon leer

        inner_dictio.append(dict_inner)
        edge_val.append((s, e))
    # combine the extracted transects as dicts to the previously inputted outer-dict.
    dict_outer = dict(zip(edge_val, inner_dictio))
    return dict_outer


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def do_analysis(year):
    if year == 2009:
        H, coord_dict = read_graph(edgelist_loc='./data/a_2009/arf_graph_2009.edgelist',
                                   coord_dict_loc='./data/a_2009/arf_graph_2009_node-coords.npy')

        img1 = Image.open('./data/a_2009/arf_dtm_2009.tif')
        img1 = np.array(img1)
        # extract transects of 9 meter width (trough_width*2 + 1 == 9)
        trough_width = 4
        transect_dict = get_transects(H, img1, trough_width)
        save_obj(transect_dict, './data/b_2019/arf_transect_dict_2009')
    elif year == 2019:
        H, coord_dict = read_graph(edgelist_loc='./data/b_2019/arf_graph_2019.edgelist',
                                   coord_dict_loc='./data/b_2019/arf_graph_2019_node-coords.npy')

        img1 = Image.open('./data/b_2019/arf_dtm_2019.tif')
        img1 = np.array(img1)
        # extract transects of 9 meter width (trough_width*2 + 1 == 9)
        trough_width = 4
        transect_dict = get_transects(H, img1, trough_width)
        save_obj(transect_dict, './data/b_2019/arf_transect_dict_2019')
    else:
        print('we do not have data from this year. please select a different year (i.e., 2009, 2019).')

if __name__ == '__main__':
    startTime = datetime.now()

    do_analysis(2009)
    do_analysis(2019)

    print(datetime.now() - startTime)
    plt.show()

#!/usr/bin/env python

import numpy as np
from PIL import Image
import sys
import networkx as nx
import pickle
from affine import Affine
from osgeo import gdal_array
from osgeo import gdal
from datetime import datetime

# np.set_printoptions(threshold=sys.maxsize)


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
    G = nx.read_edgelist(edgelist_loc, data=True, create_using=nx.DiGraph())
    coord_dict = np.load(coord_dict_loc, allow_pickle=True).item()

    return G, coord_dict


def get_geo_extent(geo_dtm):
    geoTransform = geo_dtm.GetGeoTransform()
    print(geoTransform)
    minx = geoTransform[0]
    maxy = geoTransform[3]
    maxx = minx + geoTransform[1] * geo_dtm.RasterXSize
    miny = maxy + geoTransform[5] * geo_dtm.RasterYSize
    geo_dtm = None
    return int(minx), int(miny), int(maxx), int(maxy)


def retrieve_pixel_coords(geo_coord, data_source):
    """Return floating-point value that corresponds to given point."""
    x, y = int(geo_coord[0]), int(geo_coord[1])
    forward_transform = Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform
    px, py = reverse_transform * (x, y)
    pixel_coord = int(px), int(py)

    data_array = np.array(data_source.GetRasterBand(1).ReadAsArray())
    return pixel_coord


def retrieve_world_coords(pixel_coord, data_source):
    """Return floating-point value that corresponds to given point."""
    x, y = int(pixel_coord[0]), int(pixel_coord[1])
    forward_transform = Affine.from_gdal(*data_source.GetGeoTransform())
    px, py = forward_transform * (x, y)
    world_coord = int(px), int(py)
    return world_coord


def get_transects(graph, dtm_np, dtm, width):
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
    :param dtm_np: np.array of the DTM image
    :param dtm: DTM with georeference
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
        # print(ps)
        # for i in ps:
        #     print(i)
        #     retrieve_pixel_coords(i, dtm)
        for i in range(1, len(ps)-1):
            water = False
            keys_inner.append((int(ps[i][0]), int(ps[i][1])))  # x and y coordinates of each of my current edge pixels
            px_current_rw = (int(ps[i][0]), int(ps[i][1]))  # coords of pixel p
            px_current = retrieve_pixel_coords(px_current_rw, dtm)
            px_prev_rw = (int(ps[i-1][0]), int(ps[i-1][1]))  # coords of pixel p-1
            px_prev = retrieve_pixel_coords(px_prev_rw, dtm)
            px_subs_rw = (int(ps[i+1][0]), int(ps[i+1][1]))  # coords of pixel p+1
            px_subs = retrieve_pixel_coords(px_subs_rw, dtm)

            # subset square is needed for diagonal calculation --> subset_dtm_np
            subset_dtm_np = dtm_np[px_current[0] - width:px_current[0] + width + 1, px_current[1] - width:px_current[1] + width + 1]

            # make sure to not consider any cases at the border of the image
            # to avoid only partial transects
            # if minx + width < px_current[0] < maxx-width and miny < px_current[1] < maxy-width:
            if width < px_current[0] < dtm_np.shape[0]-width and width < px_current[1] < dtm_np.shape[1]-width:
                # now consider the five possible scenarios (see publication for details)
                # if p, p+1 and p-1 are in the same row (so x-value is the same), do:
                # this is scenario a; transect is vertical
                if px_prev[0] == px_current[0] == px_subs[0]:
                    trans_start = px_current[0] - width  # where the transect begins
                    trans_end = px_current[0] + width  # where the transect ends
                    cons = px_current[1]  # and the fixed y-value/column
                    trans_start_rw = px_current_rw[0] - width  # where the transect begins
                    trans_end_rw = px_current_rw[0] + width  # where the transect ends
                    cons_rw = px_current_rw[1]  # and the fixed y-value/column

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start_rw, trans_end_rw+1))
                    y = [cons_rw]*len(x)
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
                    cons = px_current[1]  # and the fixed y-value/column
                    trans_start_rw = px_current_rw[0] - width  # where the transect begins
                    trans_end_rw = px_current_rw[0] + width  # where the transect ends
                    cons_rw = px_current_rw[1]  # and the fixed y-value/column

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start_rw, trans_end_rw+1))
                    y = [cons_rw]*len(x)
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
                    trans_start_rw = px_current_rw[0] - width  # where the transect begins
                    trans_end_rw = px_current_rw[0] + width  # where the transect ends
                    cons_rw = px_current_rw[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start_rw, trans_end_rw+1))
                    y = [cons_rw]*len(x)
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
                    trans_start_rw = px_current_rw[0] - width  # where the transect begins
                    trans_end_rw = px_current_rw[0] + width  # where the transect ends
                    cons_rw = px_current_rw[1]  # and the fixed y-value/row

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[trans_start:trans_end + 1, cons]
                    x = list(range(trans_start_rw, trans_start_rw+1))
                    y = [trans_start_rw]*len(x)
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
                    trans_start_rw = px_current_rw[1] - width  # where the transect begins
                    trans_end_rw = px_current_rw[1] + width  # where the transect ends
                    cons_rw = px_current_rw[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    # print(f'cons: {cons}')
                    # print(f'trans_start: {trans_start}')
                    # print(f'trans_end: {trans_end}')
                    transect_heights = dtm_np[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start_rw, trans_end_rw+1))
                    x = [cons_rw]*len(y)
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
                    trans_start_rw = px_current_rw[1] - width  # where the transect begins
                    trans_end_rw = px_current_rw[1] + width  # where the transect ends
                    cons_rw = px_current_rw[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start_rw, trans_end_rw+1))
                    x = [cons_rw]*len(y)
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
                    trans_start_rw = px_current_rw[1] - width  # where the transect begins
                    trans_end_rw = px_current_rw[1] + width  # where the transect ends
                    cons_rw = px_current_rw[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start_rw, trans_end_rw+1))
                    x = [cons_rw]*len(y)
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
                    trans_start_rw = px_current_rw[1] - width  # where the transect begins
                    trans_end_rw = px_current_rw[1] + width  # where the transect ends
                    cons_rw = px_current_rw[0]  # and the fixed x-value/col

                    # extract the height information from the DEM at the transect locations
                    transect_heights = dtm_np[cons, trans_start:trans_end + 1]
                    y = list(range(trans_start_rw, trans_end_rw+1))
                    x = [cons_rw]*len(y)
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
                    transect_heights = subset_dtm_np.diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] - width,  px_current[1] + width + 1))
                    x_rw = list(range(px_current_rw[0] - width, px_current_rw[0] + width + 1))
                    y_rw = list(range(px_current_rw[1] - width, px_current_rw[1] + width + 1))

                    transect_loc = list(zip(x_rw, y_rw))
                    t_type = "diagonal"  # transect is ul to lr
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] < px_current[0] < px_subs[0] and px_prev[1] > px_current[1] > px_subs[1]:
                    transect_heights = subset_dtm_np.diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] - width,  px_current[1] + width + 1))
                    x_rw = list(range(px_current_rw[0] - width, px_current_rw[0] + width + 1))
                    y_rw = list(range(px_current_rw[1] - width, px_current_rw[1] + width + 1))

                    transect_loc = list(zip(x_rw, y_rw))
                    t_type = "diagonal"  # transect is ul to lr
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] < px_current[0] < px_subs[0] and px_prev[1] < px_current[1] < px_subs[1]:
                    transect_heights = np.fliplr(subset_dtm_np).diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] + width,  px_current[1] - (width + 1), -1))
                    x_rw = list(range(px_current_rw[0] - width, px_current_rw[0] + width + 1))
                    y_rw = list(range(px_current_rw[1] + width, px_current_rw[1] - (width + 1), -1))

                    transect_loc = list(zip(x_rw, y_rw))
                    t_type = "diagonal"  # transect is ll to ur
                    t_cat = "c"
                    # if more then half of the transect pixels have the same height value, we assume that there is water
                    # in the trough.
                    if len(set(transect_heights)) <= width:
                        water = True
                    values_inner.append([transect_heights, transect_loc, t_type, t_cat, water])
                # this is scenario c; transect is diagonal
                elif px_prev[0] > px_current[0] > px_subs[0] and px_prev[1] > px_current[1] > px_subs[1]:
                    transect_heights = np.fliplr(subset_dtm_np).diagonal()
                    x = list(range(px_current[0] - width, px_current[0] + width + 1))
                    y = list(range(px_current[1] + width,  px_current[1] - (width + 1), -1))
                    x_rw = list(range(px_current_rw[0] - width, px_current_rw[0] + width + 1))
                    y_rw = list(range(px_current_rw[1] + width, px_current_rw[1] - (width + 1), -1))

                    transect_loc = list(zip(x_rw, y_rw))
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
                # print("ended up with this wrong edge... image border?")
                # # print(s, e)  # these are the border cases, but they still have some transects, so all good
                # print(px_current[0], px_current[1])
        # now recombine all elements to the inner transect dict
        # print(keys_inner, values_inner)
        dict_inner = dict(zip(keys_inner, values_inner))  # values_inner ist auch schon leer

        inner_dictio.append(dict_inner)
        edge_val.append((s, e))
    # combine the extracted transects as dicts to the previously inputted outer-dict.
    dict_outer = dict(zip(edge_val, inner_dictio))
    # print(inner_dictio)
    return dict_outer


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def do_analysis(edgelistFile, npyFile, dtmTifFile, year):
    H, coord_dict = read_graph(edgelist_loc=edgelistFile, coord_dict_loc=npyFile)

    dtm = gdal.Open(dtmTifFile)
    dtm_np = gdal_array.LoadFile(dtmTifFile)
    # extract transects of 9 meter width: (trough_width*2 + 1 == 9)
    trough_width = 4
    transect_dict = get_transects(H, dtm_np, dtm, trough_width)
    save_obj(transect_dict, 'arf_transect_dict_'+ year)


if __name__ == '__main__':
    startTime = datetime.now()

    # evtl. Fehler hier wegen der Änderungen
    edgelistFile = sys.argv[3]
    npyFile = sys.argv[2]
    dtmTifFile = sys.argv[1]
    year = npyFile.split(".")[0].split("_")[2]

    version = sys.argv[4]

    year = ''

    if version == '1':
        year = npyFile.split(".")[0].split("_")[2]
    elif version == '2':
        year = npyFile.split(".")[0][12:]

    # edgelistFile = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_graph_2009.edgelist'
    # npyFile = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_graph_2009_node-coords.npy'
    # dtmTifFile = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/arf_dtm_2009.tif'

    do_analysis(edgelistFile, npyFile, dtmTifFile, year)

    print(datetime.now() - startTime)

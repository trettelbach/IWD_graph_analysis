#!/usr/bin/env python

import cv2
import numpy as np
from PIL import Image
import sys
import scipy
import scipy.ndimage
from skimage.morphology import skeletonize, skeletonize_3d
from scipy.ndimage.morphology import generate_binary_structure
import sknw
import networkx as nx
from scipy import ndimage
from osgeo import gdal
from osgeo import gdal_array
from affine import Affine
from datetime import datetime

startTime = datetime.now()

np.set_printoptions(threshold=sys.maxsize)


def read_data(img):
    ''' helper function to make reading in DTMs easier '''
    img1 = Image.open(img)
    img1 = np.array(img1)
    return img1


def scale_data(img):
    ''' scale the image to be between 0 and 255 '''
    img_orig = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_orig


def convert_to_int(img):
    '''convert to 8bit int'''
    img = img.astype(int)
    img = img.astype('uint8')
    return img


def detrend_dtm(dtm, trend_size):
    ''' detrend the DTM image based on a filter
    of size trend_size.
    returns microtopography of DTM'''
    subset = dtm[:, :]
    reg_trend = ndimage.uniform_filter(subset, size=trend_size)
    microtop = subset - reg_trend
    microtop = scale_data(microtop)
    microtop = convert_to_int(microtop)
    return microtop


def eliminate_small_clusters(in_img, cluster_size):
    ''' determine the number of distinct features.
    Prepares for then eliminating those clusters with
    < n pixels (to remove potential noise)
    '''
    s = generate_binary_structure(2, 2)

    # labeled_array give back the same array with each pixel in a cluster getting the same number
    # but each cluster getting different numbers
    # num_features returns the number of clusters in the array
    labeled_array, num_features = scipy.ndimage.measurements.label(in_img, structure=s)

    # get value assigned to all pixels in this cluster and number of pixels per cluster
    # print(cluster_sizes[0]) --> value assigned to all pixels in this cluster
    # print(cluster_sizes[1]) --> number of pixels per cluster
    cluster_sizes = np.unique(labeled_array, return_index=False, return_inverse=False, return_counts=True)

    # initialize empty array for setting troughs and non-troughs
    result = np.zeros_like(labeled_array)

    # creates list, that will read True for any clusters with more than x pixels
    trough_bool = []

    for i in cluster_sizes[1]:
        if i > cluster_size:
            trough_bool.append(True)
        else:
            trough_bool.append(False)

    # for loop going through all pixels of the image 'labeled_array'
    # and writing information in 'results'
    for i in range(labeled_array.shape[0]):
        for j in range(labeled_array.shape[1]):
            # if it's larger than cluster_size, make it 1
            if trough_bool[labeled_array[i][j]] and labeled_array[i][j] != 0:
                result[i][j] = 1
            else:
                result[i][j] = 0
    return result


def make_directed(graph, dtm):
    """ convert graph from nx.Graph()
    to nx.DiGraph() - for each edge (u, v)
    an edge (v, u) is generated.
    Then remove all self-looping edges
    (don't exist in nature) and upward
    sloped edges (water only runs downhill)

    :param graph: graph generated from
    skeleton of class nx.Graph
    :param dtm: original digital elevation
    model (same extent as detrended image,
    as we're working with pixel indices,
    not spatial coordinates
    :return G_d: a directed graph of
    class nx.DiGraph() with only even or
    downward slope directed edges.
    """
    # we need a helper graph, because we cant remove edges from
    # the graph while iterating through it
    G_help = graph.to_directed()
    G_d = graph.to_directed()
    for (s, e) in G_help.edges():
        # remove all self-looping edges. they don't make any real world sense...
        if s == e:
            G_d.remove_edge(s, e)

        # now remove all (s, e) edges, where s downslope of e
        # (so remove directed edges that would flow upwards...)
        elev_start = dtm[int(G_help.nodes()[s]['o'][0]), int(G_help.nodes()[s]['o'][1])]
        elev_end = dtm[int(G_help.nodes()[e]['o'][0]), int(G_help.nodes()[e]['o'][1])]
        if elev_start < elev_end:
            G_d.remove_edge(s, e)
    return G_d


def get_node_coord_dict(graph, fwd):
    ''' create dictionary with node ID as key
    and node coordinates as values

    :param graph: nx.DiGraph
    :return dictionary:
    dictionary with Node IDs as keys,
    and pixel coordinates of the nodes as values.
    '''
    nodes = graph.nodes()
    # get pixel coordinates of nodes --> ps
    ps = np.array([nodes[i]['o'] for i in nodes])
    print(ps)
    for i in ps:
        # print(i)
        # print('0000')
        # tfrm = fwd * (i[0], i[1])
        tfrm = fwd * (i[1], i[0])
        i[0] = tfrm[0]
        i[1] = tfrm[1]
    print(ps)
    # get node ID --> keys
    keys = list(range(len(nodes)))
    keys_str = []
    for i in keys:
        keys_str.append(str(i))
    keys = keys_str
    values = ps.tolist()
    dictionary = dict(zip(keys, values))
    return dictionary


def save_graph_with_coords(graph, dict, location):
    ''' save graph as edgelist to disk
    and coords for nodes as dictionary

    :param graph: nx.DiGraph representing the
    trough network
    :param dict:
    :return NA: function just for saving
    '''
    # save and write Graph as list of edges
    # edge weight 'weight' stores the actual length of the trough in meter
    nx.write_edgelist(graph, location + '.edgelist', data=True)

    # and save coordinates of graph as npy dict to disk
    fname = location + '_node-coords'
    print("fname: " + fname)
    np.save(fname, dict)


def write_geotiff(out_ds_path, arr, in_ds):
    ''' takes an np.array with pixel coordinates and
    gives it the projection of another raster.
    np.array must have same extent as georeferenced
    raster.

    :param out_ds_path: string of path and filename
    where to save the newly georeferenced raster (tif).
    :param arr: the array to georeference
    :param in_ds: the already georeferenced dataset
    that serves as reference for the arr to geore-
    ference.
    :return NA: function just for saving
    '''
    if arr.dtype == np.float32:
        arr_type = gdal.GDT_Float32
    else:
        arr_type = gdal.GDT_Int32

    driver = gdal.GetDriverByName("GTiff")
    out_ds = driver.Create(out_ds_path, arr.shape[1], arr.shape[0], 1, arr_type)
    print(in_ds.GetProjection())
    out_ds.SetProjection(in_ds.GetProjection())
    out_ds.SetGeoTransform(in_ds.GetGeoTransform())
    band = out_ds.GetRasterBand(1)
    band.WriteArray(arr)
    band.FlushCache()


def get_graph_from_dtm(raster_ds_path):
    ''' takes a georeferneced digital terrain
    model and with some image processing extracts
    the graph of the polygonal trough networks in
    the landscape.

    :param raster_ds_path: string of path and filename
    of a gereferenced DTM showing permafrost tundra
    landscapes characterized by polygonal patterned
    ground. High-centered polygons only.
    :return H: nx.DiGraph with nodes representing the
    trough intersections and edges represetning the
    troughs. Each edge further has information on the
    exact course of the trough in pixels/
    :return dictio: dictionary containing information
    on the edge parameters of H.
    '''
    # read in digital terrain model. once as georeferenced
    # raster, once as spatial-less np.array.
    dtm = gdal.Open(raster_ds_path)
    dtm_np = gdal_array.LoadFile(raster_ds_path)

    # detrend the image to return microtopographic image only
    img_det = detrend_dtm(dtm_np, 16)
    # # save microtopographic image for later use
    write_geotiff('arf_microtopo_2009.tif', img_det, dtm)

    # doing adaptive thresholding on the input image
    thresh2 = cv2.adaptiveThreshold(img_det, img_det.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    133, 11)
    thresh_unclustered = eliminate_small_clusters(thresh2, 15)

    # erode and dilate the features to deal with white noise.
    kernel = np.ones((3, 3), np.uint8)
    its = 2
    for i in range(1):
        img = cv2.dilate(np.float32(thresh_unclustered), kernel, iterations=its)
        closed = cv2.erode(img, kernel, iterations=its)
    # print(closed)

    # prepare for both possible skeletonization algorithms
    # zhang = skeletonize(img)
    lee = skeletonize_3d(img)

    img_skel = lee

    skel_clu_elim_25 = eliminate_small_clusters(img_skel, 10)

    #write_geotiff('skel_test_2009.tif',
   #      skel_clu_elim_25, dtm)

    # build graph from skeletonized image
    G = sknw.build_sknw(skel_clu_elim_25, multi=False)

    for (s, e) in G.edges():
        G[s][e]['pts'] = G[s][e]['pts'].tolist()

    geot = dtm.GetGeoTransform()
    print(geot)
    fwd = Affine.from_gdal(*geot)
    G_copy = G.copy()

    # transform the pixel coordinates to lat/lon for geospatialness
    for (s, e) in G_copy.edges():
        for i in G[s][e]['pts']:
            tfrm = fwd * (i[0], i[1])
            i[0] = tfrm[0]
            i[1] = tfrm[1]
        # print(G[s][e]['pts'])

    # and make it a directed graph, since water only flows downslope
    # flow direction is based on elevation information of DTM heights
    H = make_directed(G, dtm_np)

    # save graph and node coordinates
    dictio = get_node_coord_dict(H, fwd)

    save_graph_with_coords(H, dictio, 'arf_graph_2009')

    dtm = None
    return H, dictio


if __name__ == '__main__':
    # @Jonathan: hier will ich nun entweder eine einzelne file analysieren,
    # oder eben eine liste oder alle dateien aus einem directory.
    # wie mache ich das am besten mit sys.argv?
    raster_ds_path = sys.argv[1]

    # raster_ds_path = r'E:\02_macs_fire_sites\00_working\03_code_scripts\IWD_graph_analysis\data\arf_dtm_2009.tif'
    H, dictio = get_graph_from_dtm(raster_ds_path)

    # print time needed for script execution
    print(datetime.now() - startTime)

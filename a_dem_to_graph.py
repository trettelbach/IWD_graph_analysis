from copy import copy, deepcopy
import cv2
import numpy as np
from PIL import Image
import sys
import scipy
import scipy.ndimage
from skimage.morphology import skeletonize, medial_axis, skeletonize_3d
from scipy.ndimage.morphology import generate_binary_structure
import sknw
import matplotlib.pyplot as plt
import networkx as nx
from scipy import ndimage
from networkx.readwrite import json_graph
import matplotlib
# import simplejson as json
from datetime import datetime

startTime = datetime.now()

np.set_printoptions(threshold=sys.maxsize)

def read_data(img):
    ''' helper function to make reading in DEMs easier '''
    img1 = Image.open(img)
    img1 = np.array(img1)
    return img1


def scale_data(img):
    ''' scale the image to be between 0 and 255 '''
    img_orig = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return img_orig


def int_conversion(img):
    '''convert to 8bit int'''
    img = img.astype(int)
    img = img.astype('uint8')
    return img


def detrender(dem, trend_size):
    ''' detrend the DEM image based on a filter
    of size trend_size.
    returns microtopography of DEM'''
    subset = dem[:, :]
    reg_trend = ndimage.uniform_filter(subset, size=trend_size)
    microtop = subset - reg_trend
    microtop = scale_data(microtop)
    microtop = int_conversion(microtop)
    return microtop


def small_cluster_elim(in_img, cluster_size):
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


def make_directed(graph, dem):
    """ convert graph from nx.Graph()
    to nx.DiGraph() - for each edge (u, v)
    an edge (v, u) is generated.
    Then remove all self-looping edges
    (don't exist in nature) and upward
    sloped edges (water only runs downhill)

    :param graph: graph generated from
    skeleton of class nx.Graph
    :param dem: original digital elevation
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
        elev_start = dem[int(G_help.nodes()[s]['o'][0]), int(G_help.nodes()[s]['o'][1])]
        elev_end = dem[int(G_help.nodes()[e]['o'][0]), int(G_help.nodes()[e]['o'][1])]
        if elev_start < elev_end:
            G_d.remove_edge(s, e)
    return G_d


def get_node_coord_dict(graph):
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
    np.save(fname, dict)


def make_process_plot(img_orig, img_det, thresh2, thresh_unclustered, closed, img_skel, skel_clu_elim_25, skel_transp,
                      save_loc):
    ''' plot the 7 substeps of the
    analysis in one plot
    '''
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 10), sharex='all', sharey='all')

    # DTM
    axs[0, 0].imshow(img_orig, cmap='gray')

    # detrended DEM
    axs[0, 1].imshow(img_det, cmap='gray')

    # binarized segmentation
    axs[1, 0].imshow(thresh2, cmap='binary')

    # unclustered seg
    axs[1, 1].imshow(thresh_unclustered, cmap='binary')

    # morph. closed
    axs[2, 0].imshow(closed, cmap='binary')

    # skeleton
    axs[2, 1].imshow(img_skel, cmap='binary')

    # unclustered skel
    axs[3, 0].imshow(skel_clu_elim_25, cmap='binary')

    # skel. on detr. DEM
    axs[3, 1].imshow(img_det, cmap='gray')
    axs[3, 1].imshow(skel_transp)

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.savefig(save_loc, dpi=900, bbox_inches='tight', pad_inches=0)


def do_analysis(year):
    its = 2
    # 2009
    if year == 2009:
        img_orig = read_data('./data/a_2009/arf_dtm_2009.tif')
        its = 1

    # 2019
    elif year == 2019:
        img_orig = read_data('./data/b_2019/arf_dtm_2019.tif')
        its = 2

    # detrend the image to return microtopographic image only
    img_det = detrender(img_orig, 16)

    # doing adaptive thresholding on the input image
    thresh2 = cv2.adaptiveThreshold(img_det, img_det.max(), cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                    133, 11)
    thresh_unclustered = small_cluster_elim(thresh2, 15)

    # erode and dilate the features to deal with white noise.
    kernel = np.ones((5, 5), np.uint8)
    for i in range(1):
        img = cv2.dilate(np.uint8(thresh_unclustered), kernel, iterations=its)
        closed = cv2.erode(img, kernel, iterations=1)

    # prepare for both possible skeletonization algorithms
    zhang = skeletonize(img)
    lee = skeletonize_3d(img)

    img_skel = lee

    # then eliminate small clusters < 25 pixels total (aka noise)
    skel_clu_elim_25 = small_cluster_elim(img_skel, 25)
    print(skel_clu_elim_25)

    im = Image.fromarray(skel_clu_elim_25)

    # make a transparent raster with only trough pixels in red.
    skel_transp = np.zeros((skel_clu_elim_25.shape[0], skel_clu_elim_25.shape[1], 4))
    for i in range(skel_clu_elim_25.shape[0]):
        for j in range(skel_clu_elim_25.shape[1]):
            if skel_clu_elim_25[i, j] == 1:
                skel_transp[i, j, 0] = 255
                skel_transp[i, j, 3] = 255

    # build graph from skeletonized image
    G = sknw.build_sknw(skel_clu_elim_25, multi=False)

    # need to avoid np.arrays - so we convert it to a list
    for (s, e) in G.edges():
        G[s][e]['pts'] = G[s][e]['pts'].tolist()

    # and make it a directed graph, since water only flows downslope
    # flow direction is based on elevation informaiton of DEM heights
    dem = img_orig
    H = make_directed(G, dem)

    # save graph and node coordinates
    dictio = get_node_coord_dict(H)

    if year == 2009:
        save_graph_with_coords(H, dictio, './data/a_2009/arf_graph_2009')
    elif year == 2019:
        save_graph_with_coords(H, dictio, './data/b_2019/arf_graph_2019')

    # make_process_plot(img_orig, img_det, thresh2, thresh_unclustered, closed, img_skel, skel_clu_elim_25,
    #                  skel_transp,
    #                  save_loc='D:/test19.png')

    plt.figure(dpi=300)
    plt.imshow(img_det, cmap='bone')
    plt.imshow(skel_transp)
    plt.axis('off')
    # plt.savefig('./data/arf_skel_on_dtm_2019', bbox_inches='tight')
    return H, dictio


if __name__ == '__main__':
    plt.figure()
    H_09, dictio_09 = do_analysis(2009)
    H_19, dictio_19 = do_analysis(2019)

    # print time needed for script execution
    print(datetime.now() - startTime)
    plt.show()

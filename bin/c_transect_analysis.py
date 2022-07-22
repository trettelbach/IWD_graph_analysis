#!/usr/bin/env python

import sys
import pickle
import numpy as np
from PIL import Image
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from datetime import datetime
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

startTime = datetime.now()
np.set_printoptions(threshold=sys.maxsize)


def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def read_data(img1):
    ''' helper function to make reading in DEMs easier '''
    # this is the original DEM
    if img1 == "original":
        # img1 = Image.open('D:/01_anaktuvuk_river_fire/00_working/01_processed-data/00_study-area'
        #                   '/li-dem_1m_sa3_fill.tif')
        img1 = Image.open('D:/01_anaktuvuk_river_fire/00_working/01_processed-data/00_study-area/bens_data'
                          '/ben_2009_DTM_1m_small-sa.tif')
        img1 = np.array(img1)

    # this is the microtopo image:
    if img1 == "detrended":
        # img1 = Image.open('D:/01_anaktuvuk_river_fire/00_working/01_processed-data/02_microtopography'
        #                   '/awi_2019_DTM_1m_reproj_300x300_02_microtopo_16m.tif')
        img1 = Image.open("D:/01_anaktuvuk_river_fire/00_working/01_processed-data/02_microtopography/"
                          "ben_2009_DTM_1m_small-sa_detrended_16m.tif")
        img1 = np.array(img1)
    return img1


def inner(key, val, out_key):
    ''' fits a gaussian to every transect
    height profile and adds transect parameters
    to the dictionary.

    :param key: coords of trough pixel
    (determines center of transect)
    :param val: list of transect heights,
    coords, and directionality/type
    :param out_key: current edge with (s, e)
    :return val: updated val with:
    - val[5] = fwhm_gauss --> transect width
    - val[6] = mean_gauss --> transect depth
    - val[7] = cod_gauss --> r2 of fit
    '''

    # implement the gaussian function
    def my_gaus(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    # check if there's a transect to fit in the first place
    # (some transects at the image edge/corner might be empty) --> but there are none
    # print(val[0])
    if len(val[0]) != 0:
        # flip the transect along x-axis to be able to fit the Gaussian
        data = val[0] * (-1) + np.max(val[0])
        N = len(data)  # number of data points (corresponds to width*2 + 1)

        # diagonal transects are sqrt(2) times longer than straight transects
        if val[2] == "diagonal":
            t = np.linspace(0, (len(data)) * np.sqrt(2), N)
        else:
            t = np.linspace(0, len(data) - 1, N)

        # provide initial guesses for the mean and sigma for fitting
        mean = np.argmax(data)  # mean is estimated to be at the maximum point of the flipped transect
        # (lowest point within the trough)
        sigma = np.sqrt(sum(data * (t - mean) ** 2) / N) + 1  # estimate for sigma is determined via the underlying data
        # + 1 to avoid division by 0 for flat transects

        # now fit the Gaussian & raise error for those that can't be fitted
        try:
            gauss_fit = curve_fit(my_gaus, t, data, p0=[1, mean, sigma], maxfev=500000,
                                  bounds=[(-np.inf, -np.inf, 0.01), (np.inf, np.inf, 8.5)])
        except RuntimeError:
            print('RuntimeError is raised with edge: {0} coords {1} and elevations: {2}'.format(out_key, key, val))
            # pass

        try:
            # recreate the fitted curve using the optimized parameters
            data_gauss_fit = my_gaus(t, *gauss_fit[0])

            # and finally get depth and width and r2 of fit for adding to original dictionary (val)
            max_gauss = np.max(data_gauss_fit)
            fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * abs(gauss_fit[0][2])
            cod_gauss = r2_score(data, data_gauss_fit)
            # append the parameters to val
            val.append(fwhm_gauss)
            val.append(max_gauss)
            val.append(cod_gauss)

            # plotting=True
            # if key[0]==15 and key[1]==610:
            #     plt.plot(t, data, '+:', label='DTM elevation', color='darkslategrey')
            #     plt.plot(t, data_gauss_fit, color='lightseagreen',
            #              label='fitted Gaussian')
            #     # , d={0}, w={1}, r2={2}'.format(round(max_gauss, 2),
            #     #                                                                     round(fwhm_gauss, 2),
            #     #                                                                     round(cod_gauss, 2)
            #     plt.legend(frameon=False)
            #     plt.ylabel("depth below ground [m]")
            #     plt.xlabel("transect length [m]")
            #     plt.xticks(np.arange(9), np.arange(1, 10))
            #     plt.text(0, 0.25, f'trough width: {round(fwhm_gauss, 2)} m', fontsize=8)
            #     plt.text(0, 0.235, f'trough depth: {round(max_gauss, 2)} m', fontsize=8)
            #     plt.text(0, 0.22, f'$r^2$ of fit: {round(cod_gauss, 2)}', fontsize=8)
            #     # plt.title("direction: {0}, category: {1}".format(val[2], val[3]))
            #     plt.savefig('./figures/fitted_to_coords_{0}_{1}.png'.format(key[0], key[1]), dpi=300)
            #     plt.close()
        except:
            # bad error handling:
            if val[4]:
                print("a water-filled trough can't be fitted: edge: {}".format(out_key))
            else:
                print("something seriously wrong")
    else:
        # print(val)
        print("boooooo")

    return val


def outer(out_key, inner_dict):
    ''' iterate through all transects of a
    single trough and send to inner()
    where gaussian will be fitted.

    :param out_key: current edge with (s, e)
    :param inner_dict: dict of transects with:
    - inner_keys: pixel-coords of trough pixels (x, y)
    inbetween (s, e).
    - inner_values: list with transect coordinates
    and info on directionality/type
    :return inner_dict: updated inner_dict with old
    inner_values + transect width, height, r2 in val
    '''
    all_keys = []
    all_vals_upd = []
    # iterate through all transects of a trough
    for key, val in inner_dict.items():
        try:
            # fit gaussian to all transects
            val_upd = inner(key, val, out_key)
            all_keys.append(key)
            all_vals_upd.append(val_upd)
        except ValueError as err:
            print('{0} -- {1}'.format(out_key, err))
    # recombine keys and vals to return the updated dict
    inner_dict = dict(zip(all_keys, all_vals_upd))
    return inner_dict


def fit_gaussian_parallel(dict_soil):
    '''iterate through edges of the graph (in dict
    form) and send each trough to a free CPU core
    --> prepare fitting a Gaussian function
    to the extracted transects in dict_soil
    for parallel processing: each trough will
    be handled by a single CPU core, but different
    troughs can be distributed to multiple cores.

    :param dict_soil: a dictionary with
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
    :return dict_soil2: updated dict soil
    same as dict_soil with added:
    - inner_values:
        - val[5] = fwhm_gauss --> transect width
        - val[6] = mean_gauss --> transect depth
        - val[7] = cod_gauss --> r2 of fit
    '''
    all_outer_keys = []
    # parallelize into n_jobs different jobs/CPU cores
    out = Parallel(n_jobs=20)(delayed(outer)(out_key, inner_dict) for out_key, inner_dict in dict_soil.items())
    # get all the outer_keys
    for out_key, inner_dict in dict_soil.items():
        all_outer_keys.append(out_key)
    # and recombine them with the updated inner_dict
    dict_soil2 = dict(zip(all_outer_keys, out))
    print(dict_soil2)
    return dict_soil2


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def get_trough_avgs_gauss(transect_dict_fitted):
    ''' gather all width/depth/r2 parameters of
    each transect and compute mean/median
    parameter per trough. Add mean/median per
    trough to the dict.
    this part is mainly preparation for the
    later network_analysis(.py).

    :param transect_dict_fitted:
    :return mean_trough_params: a copy of the
    transect_dict_fitted with added mean trough
    parameters to the outer dict as values.
    '''
    mean_trough_params = {}
    empty_edges = []

    # print(transect_dict_fitted)
    # iterate through all edges/troughs
    for edge, trough in transect_dict_fitted.items():
        num_trans_tot = len(trough)  # get the total number of transects in one edge/trough
        gaus_width_sum = []
        gaus_depth_sum = []
        gaus_r2_sum = []
        num_trans_cons = 0
        water = 0
        # check if an edge/trough is empty
        if trough != {}:
            # then iterate through all transects of the current edge/trough
            for coords, trans in trough.items():
                # print(coords)
                # filter out all transects that:
                # a) are not between 0 m and 15 m in width (unrealistic values)
                # b) have been fitted with r2 <= 0.8
                # c) likely have water present
                if not isinstance(trans, list):
                    pass
                # now count number of water-filled transects per trough
                elif trans[4]:
                    water += 1
                # pass
                elif len(trans[0]) != 0 and 0 < trans[5] < 15 and trans[7] > 0.8 and not trans[4]:
                    # append the parameters from "good" transects to the lists
                    gaus_width_sum.append(trans[5])
                    gaus_depth_sum.append(trans[6])
                    gaus_r2_sum.append(trans[7])
                    num_trans_cons += 1

            # to then calculate the mean/median for each parameter
            gaus_mean_width = np.mean(gaus_width_sum)
            gaus_median_width = np.median(gaus_width_sum)
            gaus_mean_depth = np.mean(gaus_depth_sum)
            gaus_median_depth = np.median(gaus_depth_sum)
            gaus_mean_r2 = np.mean(gaus_r2_sum)
            gaus_median_r2 = np.median(gaus_r2_sum)
            # ratio of "good" transects considered for mean/median params compared to all transects available
            perc_trans_cons = np.round(num_trans_cons / num_trans_tot, 2)
            perc_water_fill = np.round(water / len(trough), 2)
            # add all the mean/median parameters to the inner_dict
            mean_trough_params[edge] = [gaus_mean_width, gaus_median_width,
                                        gaus_mean_depth, gaus_median_depth,
                                        gaus_mean_r2, gaus_median_r2,
                                        perc_trans_cons, perc_water_fill]
        # and if the trough is empty, append the edge to the list of empty edges
        else:
            empty_edges.append(edge)
            # print(transect_dict_fitted[edge])
    # print('empty edges ({0} in total): {1}'.format(len(empty_edges), empty_edges))
    return mean_trough_params


def plot_param_hists_box_width(transect_dict_orig_fitted_09, transect_dict_orig_fitted_19):
    ''' plot and save histogram and boxplot
    of all transect widths distribution for
    two points in time and for all vs.
    filtered results.

    :param transect_dict_orig_fitted_09:
    dictionary of 2009 situation
    :param transect_dict_orig_fitted_19:
    dictionary of 2019 situation
    :return: plot with hist and boxplot
    '''
    all_widths_09 = []
    hi_widths_09 = []

    for edge, inner_dic in transect_dict_orig_fitted_09.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_widths_09.append(np.abs(trans_info[5]))
                if trans_info[7] > 0.8:
                    hi_widths_09.append(np.abs(trans_info[5]))

    all_widths_19 = []
    hi_widths_19 = []

    for edge, inner_dic in transect_dict_orig_fitted_19.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_widths_19.append(np.abs(trans_info[5]))
                if trans_info[7] > 0.8:
                    hi_widths_19.append(np.abs(trans_info[5]))

    # print(f'all widths: \t 2009: {len(all_widths_09)} \t 2019: {len(all_widths_19)}')
    # print(f'hi widths: \t 2009: {len(hi_widths_09)} \t 2019: {len(hi_widths_19)}')
    print("WIDTH")
    print("r2 > 0.8")
    print(f'median width: \t 2009: {np.median(hi_widths_09)} \t 2019: {np.median(hi_widths_19)}')
    print(f'mean width: \t 2009: {np.mean(hi_widths_09)} \t 2019: {np.mean(hi_widths_19)}')
    print(f'min width: \t 2009: {np.min(hi_widths_09)} \t 2019: {np.min(hi_widths_19)}')
    print(f'max width: \t 2009: {np.max(hi_widths_09)} \t 2019: {np.max(hi_widths_19)}')
    print(f'std width: \t 2009: {np.std(hi_widths_09)} \t 2019: {np.std(hi_widths_19)}')
    print("all r2")
    print(f'median width: \t 2009: {np.median(all_widths_09)} \t 2019: {np.median(all_widths_19)}')
    print(f'mean width: \t 2009: {np.mean(all_widths_09)} \t 2019: {np.mean(all_widths_19)}')
    print(f'min width: \t 2009: {np.min(all_widths_09)} \t 2019: {np.min(all_widths_19)}')
    print(f'max width: \t 2009: {np.max(all_widths_09)} \t 2019: {np.max(all_widths_19)}')
    print(f'std width: \t 2009: {np.std(all_widths_09)} \t 2019: {np.std(all_widths_19)}')
    print("______________________________________________________________")

    # do the plotting
    boxplotprops_09 = {'patch_artist': True,
                       'boxprops': dict(facecolor='salmon'),
                       'flierprops': dict(marker='o', markerfacecolor='salmon', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='salmon')}
    boxplotprops_19 = {'patch_artist': True,
                       'boxprops': dict(facecolor='teal'),
                       'flierprops': dict(marker='o', markerfacecolor='teal', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='teal')}
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(3, 3), dpi=300,
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 2, 6]})
    # fig.tight_layout()
    # axes[0].axis('off')

    median_09, q1_09, q3_09 = np.percentile(hi_widths_09, 50), np.percentile(hi_widths_09, 25), np.percentile(
        hi_widths_09, 75)
    median_19, q1_19, q3_19 = np.percentile(hi_widths_19, 50), np.percentile(hi_widths_19, 25), np.percentile(
        hi_widths_19, 75)

    # 2009 boxplot
    axes[0].boxplot(hi_widths_09, 1, vert=False, widths=0.5, **boxplotprops_09)
    axes[0].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    # axes[0].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    axes[0].set_yticks([])
    axes[0].set_yticklabels([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].set_ylabel('2009', weight='bold')

    # 2019 boxplot
    axes[1].boxplot(hi_widths_19, 1, vert=False, widths=0.5, **boxplotprops_19)
    axes[1].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    # axes[1].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_ylabel('2019', weight='bold')

    # histogram
    # 2009
    axes[2].hist(all_widths_09, bins=np.arange(0.0, 20.0, 0.4), range=(0, 20), histtype='step', color='peachpuff',
                 label=r"width (all)")
    axes[2].hist(hi_widths_09, bins=np.arange(0.0, 20.0, 0.4), range=(0, 20), histtype='step', color='salmon',
                 label=r"width ($r^2 > 0.8$)")
    axes[2].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9,
                    label="median = {0} m".format(np.round(median_09, 2)))
    # 2019
    axes[2].hist(all_widths_19, bins=np.arange(0.0, 20.0, 0.4), range=(0, 20), histtype='step', color='powderblue',
                 label=r"width (all)")
    axes[2].hist(hi_widths_19, bins=np.arange(0.0, 20.0, 0.4), range=(0, 20), histtype='step', color='teal',
                 label=r"width ($r^2 > 0.8$)")
    axes[2].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9,
                    label="median = {0} m".format(np.round(median_19, 2)))

    axes[2].set_ylabel('frequency')
    axes[2].set_xlabel('width [m]')
    # axes[0].set_title("Trough Widths")

    # # prepare legend
    # handles, labels = axes[2].get_legend_handles_labels()
    # # colors = ['peachpuff', 'salmon', 'salmon', 'powderblue', 'teal', 'teal']
    # # lstyles = ['-', '-', '--', '-', '-', '--']
    # # item_melting = mlines.Line2D([], [], color=colors, linestyle=lstyles, linewidth=1)
    # # handles[0] = item_melting
    # order = [2, 3, 0, 4, 5, 1]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center',
    #            bbox_to_anchor=(0.775, 0.875), ncol=1, frameon=False, fontsize=9)
    # plt.gcf().text(0.565, 0.416, r'2009', fontsize=10, weight='bold', rotation=90)
    # plt.gcf().text(0.565, 0.305, r'2019', fontsize=10, weight='bold', rotation=90)
    # # axes[0].subplots_adjust(top=0.5)
    # # plt.show()
    fig.tight_layout()
    # plt.savefig('./figures/hist_box_width.png')


def plot_param_hists_box_depth(transect_dict_orig_fitted_09, transect_dict_orig_fitted_19):
    ''' plot and save histogram and boxplot
    of all transect depths distribution for
    two points in time and for all vs.
    filtered results.

    :param transect_dict_orig_fitted_09:
    dictionary of 2009 situation
    :param transect_dict_orig_fitted_19:
    dictionary of 2019 situation
    :return: plot with hist and boxplot
    '''
    all_depths_09 = []
    hi_depths_09 = []

    for edge, inner_dic in transect_dict_orig_fitted_09.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_depths_09.append(trans_info[6])
                if trans_info[7] > 0.8:
                    hi_depths_09.append(trans_info[6])

    all_depths_19 = []
    hi_depths_19 = []

    for edge, inner_dic in transect_dict_orig_fitted_19.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_depths_19.append(trans_info[6])
                if trans_info[7] > 0.8:
                    hi_depths_19.append(trans_info[6])

    # print(f'all depths: \t 2009: {len(all_depths_09)} \t 2019: {len(all_depths_19)}')
    # print(f'hi depths: \t 2009: {len(hi_depths_09)} \t 2019: {len(hi_depths_19)}')
    # print(f'median depths: \t 2009: {np.median(hi_depths_09)} \t 2019: {np.median(hi_depths_19)}')
    # print(f'mean depths: \t 2009: {np.mean(hi_depths_09)} \t 2019: {np.mean(hi_depths_19)}')
    # print(f'min depths: \t 2009: {np.min(hi_depths_09)} \t 2019: {np.min(hi_depths_19)}')
    # print(f'max depths: \t 2009: {np.max(hi_depths_09)} \t 2019: {np.max(hi_depths_19)}')

    print("DEPTH")
    print("r2 > 0.8")
    print(f'median depths: \t 2009: {np.median(hi_depths_09)} \t 2019: {np.median(hi_depths_19)}')
    print(f'mean depths: \t 2009: {np.mean(hi_depths_09)} \t 2019: {np.mean(hi_depths_19)}')
    print(f'min depths: \t 2009: {np.min(hi_depths_09)} \t 2019: {np.min(hi_depths_19)}')
    print(f'max depths: \t 2009: {np.max(hi_depths_09)} \t 2019: {np.max(hi_depths_19)}')
    print(f'std depths: \t 2009: {np.std(hi_depths_09)} \t 2019: {np.std(hi_depths_19)}')
    print("all r2")
    print(f'median depths: \t 2009: {np.median(all_depths_09)} \t 2019: {np.median(all_depths_19)}')
    print(f'mean depths: \t 2009: {np.mean(all_depths_09)} \t 2019: {np.mean(all_depths_19)}')
    print(f'min depths: \t 2009: {np.min(all_depths_09)} \t 2019: {np.min(all_depths_19)}')
    print(f'max depths: \t 2009: {np.max(all_depths_09)} \t 2019: {np.max(all_depths_19)}')
    print(f'std depths: \t 2009: {np.std(all_depths_09)} \t 2019: {np.std(all_depths_19)}')
    print("______________________________________________________________")

    # do the plotting
    boxplotprops_09 = {'patch_artist': True,
                       'boxprops': dict(facecolor='salmon'),
                       'flierprops': dict(marker='o', markerfacecolor='salmon', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='salmon')}
    boxplotprops_19 = {'patch_artist': True,
                       'boxprops': dict(facecolor='teal'),
                       'flierprops': dict(marker='o', markerfacecolor='teal', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='teal')}
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(3, 3), dpi=300,
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 2, 6]})
    # fig.tight_layout()
    # axes[0].axis('off')

    median_09, q1_09, q3_09 = np.percentile(hi_depths_09, 50), np.percentile(hi_depths_09, 25), np.percentile(
        hi_depths_09, 75)
    median_19, q1_19, q3_19 = np.percentile(hi_depths_19, 50), np.percentile(hi_depths_19, 25), np.percentile(
        hi_depths_19, 75)

    # 2009 boxplot
    axes[0].boxplot(hi_depths_09, 1, vert=False, widths=0.5, **boxplotprops_09)
    axes[0].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    # axes[0].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    axes[0].set_yticks([])
    axes[0].set_yticklabels([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].set_ylabel('2009', weight='bold')

    # 2019 boxplot
    axes[1].boxplot(hi_depths_19, 1, vert=False, widths=0.5, **boxplotprops_19)
    axes[1].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    # axes[1].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_ylabel('2019', weight='bold')

    # histogram
    # 2009
    axes[2].hist(all_depths_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='peachpuff',
                 label=r"depth (all)")
    axes[2].hist(hi_depths_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='salmon',
                 label=r"depth ($r^2 > 0.8$)")
    axes[2].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9,
                    label="median = {0} m".format(np.round(median_09, 2)))
    # 2019
    axes[2].hist(all_depths_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='powderblue',
                 label=r"depth (all)")
    axes[2].hist(hi_depths_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='teal',
                 label=r"depth ($r^2 > 0.8$)")
    axes[2].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9,
                    label="median = {0} m".format(np.round(median_19, 2)))

    axes[2].set_ylabel('frequency')
    axes[2].set_xlabel('depth [m]')
    # axes[0].set_title("Trough Widths")

    # # prepare legend
    # handles, labels = axes[2].get_legend_handles_labels()
    # # colors = ['peachpuff', 'salmon', 'salmon', 'powderblue', 'teal', 'teal']
    # # lstyles = ['-', '-', '--', '-', '-', '--']
    # # item_melting = mlines.Line2D([], [], color=colors, linestyle=lstyles, linewidth=1)
    # # handles[0] = item_melting
    # order = [2, 3, 0, 4, 5, 1]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center',
    #            bbox_to_anchor=(0.775, 0.875), ncol=1, frameon=False, fontsize=9)
    # plt.gcf().text(0.565, 0.416, r'2009', fontsize=10, weight='bold', rotation=90)
    # plt.gcf().text(0.565, 0.305, r'2019', fontsize=10, weight='bold', rotation=90)
    # # axes[0].subplots_adjust(top=0.5)
    # # plt.show()
    fig.tight_layout()
    # plt.savefig('./figures/hist_box_depth.png')


def plot_param_hists_box_cod(transect_dict_orig_fitted_09, transect_dict_orig_fitted_19):
    ''' plot and save histogram and boxplot
    of all transect r2 of fits for
    two points in time and for all vs.
    filtered results.

    :param transect_dict_orig_fitted_09:
    dictionary of 2009 situation
    :param transect_dict_orig_fitted_19:
    dictionary of 2019 situation
    :return: plot with hist and boxplot
    '''
    all_cods_09 = []
    hi_cods_09 = []
    cod_neg_09 = 0
    cod_pos_09 = 0

    for edge, inner_dic in transect_dict_orig_fitted_09.items():
        for skel_pix, trans_info in inner_dic.items():
            if trans_info[7] < 0:
                cod_neg_09 += 1
            else:
                cod_pos_09 += 1
            if -30 < trans_info[5] < 30:
                all_cods_09.append(trans_info[7])
                if trans_info[7] > 0.8:
                    hi_cods_09.append(trans_info[7])

    print(f'{(cod_neg_09 * 100) / (cod_neg_09 + cod_pos_09)} of all fits had a r2 < 0')

    all_cods_19 = []
    hi_cods_19 = []
    cod_neg_19 = 0
    cod_pos_19 = 0

    for edge, inner_dic in transect_dict_orig_fitted_19.items():
        for skel_pix, trans_info in inner_dic.items():
            if trans_info[7] < 0:
                cod_neg_19 += 1
            else:
                cod_pos_19 += 1
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_cods_19.append(trans_info[7])
                if trans_info[7] > 0.8:
                    hi_cods_19.append(trans_info[7])

    print(f'{(cod_neg_19 * 100) / (cod_neg_19 + cod_pos_19)} of all fits had a r2 < 0')

    # print(f'all r2: \t 2009: {len(all_cods_09)} \t 2019: {len(all_cods_19)}')
    # print(f'hi r2: \t 2009: {len(hi_cods_09)} \t 2019: {len(hi_cods_19)}')
    # print(f'median r2r2: \t 2009: {np.median(hi_cods_09)} \t 2019: {np.median(hi_cods_19)}')
    # print(f'mean width: \t 2009: {np.mean(hi_cods_09)} \t 2019: {np.mean(hi_cods_19)}')
    # print(f'min r2: \t 2009: {np.min(hi_cods_09)} \t 2019: {np.min(hi_cods_19)}')
    # print(f'max r2: \t 2009: {np.max(hi_cods_09)} \t 2019: {np.max(hi_cods_19)}')

    print("R2")
    print("r2 > 0.8")
    print(f'median r2: \t 2009: {np.median(hi_cods_09)} \t 2019: {np.median(hi_cods_19)}')
    print(f'mean r2: \t 2009: {np.mean(hi_cods_09)} \t 2019: {np.mean(hi_cods_19)}')
    print(f'min r2: \t 2009: {np.min(hi_cods_09)} \t 2019: {np.min(hi_cods_19)}')
    print(f'max r2: \t 2009: {np.max(hi_cods_09)} \t 2019: {np.max(hi_cods_19)}')
    print(f'std r2: \t 2009: {np.std(hi_cods_09)} \t 2019: {np.std(hi_cods_19)}')
    print("all r2")
    print(f'median r2: \t 2009: {np.median(all_cods_09)} \t 2019: {np.median(all_cods_19)}')
    print(f'mean r2: \t 2009: {np.mean(all_cods_09)} \t 2019: {np.mean(all_cods_19)}')
    print(f'min r2: \t 2009: {np.min(all_cods_09)} \t 2019: {np.min(all_cods_19)}')
    print(f'max r2: \t 2009: {np.max(all_cods_09)} \t 2019: {np.max(all_cods_19)}')
    print(f'std r2: \t 2009: {np.std(all_cods_09)} \t 2019: {np.std(all_cods_19)}')
    print("______________________________________________________________")

    # do the plotting
    boxplotprops_09 = {'patch_artist': True,
                       'boxprops': dict(facecolor='salmon'),
                       'flierprops': dict(marker='o', markerfacecolor='salmon', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='salmon')}
    boxplotprops_19 = {'patch_artist': True,
                       'boxprops': dict(facecolor='teal'),
                       'flierprops': dict(marker='o', markerfacecolor='teal', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='teal')}
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(3, 3), dpi=300,
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 2, 6]})
    # axes[0].axis('off')

    median_09, q1_09, q3_09 = np.percentile(hi_cods_09, 50), np.percentile(hi_cods_09, 25), np.percentile(
        hi_cods_09, 75)
    median_19, q1_19, q3_19 = np.percentile(hi_cods_19, 50), np.percentile(hi_cods_19, 25), np.percentile(
        hi_cods_19, 75)

    # 2009 boxplot
    axes[0].boxplot(hi_cods_09, 1, vert=False, widths=0.5, **boxplotprops_09)
    axes[0].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    # axes[0].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    axes[0].set_yticks([])
    axes[0].set_yticklabels([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].set_ylabel('2009', weight='bold')

    # 2019 boxplot
    axes[1].boxplot(hi_cods_19, 1, vert=False, widths=0.5, **boxplotprops_19)
    axes[1].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    # axes[1].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_ylabel('2019', weight='bold')

    # histogram
    # 2009
    axes[2].hist(all_cods_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='peachpuff',
                 label=r"$r^2$ (all)")
    axes[2].hist(hi_cods_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='salmon',
                 label=r"$r^2 > 0.8$")
    axes[2].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9,
                    label="median = {0}".format(np.round(median_09, 2)))
    # 2019
    axes[2].hist(all_cods_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='powderblue',
                 label=r"$r^2$ (all)")
    axes[2].hist(hi_cods_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='teal',
                 label=r"$r^2 > 0.8$")
    axes[2].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9,
                    label="median = {0}".format(np.round(median_19, 2)))

    axes[2].set_ylabel('frequency')
    axes[2].set_xlabel(r'$r^2$')
    # axes[0].set_title("Trough cods")

    # # prepare legend
    # handles, labels = axes[2].get_legend_handles_labels()
    # # colors = ['peachpuff', 'salmon', 'salmon', 'powderblue', 'teal', 'teal']
    # # lstyles = ['-', '-', '--', '-', '-', '--']
    # # item_melting = mlines.Line2D([], [], color=colors, linestyle=lstyles, linewidth=1)
    # # handles[0] = item_melting
    # order = [2, 3, 0, 4, 5, 1]
    # plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center',
    #            bbox_to_anchor=(0.295, 0.875), ncol=1, frameon=False, fontsize=9)
    # plt.gcf().text(0.205, 0.416, r'2009', fontsize=10, weight='bold', rotation=90)
    # plt.gcf().text(0.205, 0.305, r'2019', fontsize=10, weight='bold', rotation=90)
    # # axes[0].subplots_adjust(top=0.5)
    # # plt.show()
    fig.tight_layout()
    # plt.savefig('./figures/hist_box_cod.png')


def plot_legend(transect_dict_orig_fitted_09, transect_dict_orig_fitted_19):
    ''' plot and save histogram and boxplot
    of all transect depths distribution for
    two points in time and for all vs.
    filtered results.

    :param transect_dict_orig_fitted_09:
    dictionary of 2009 situation
    :param transect_dict_orig_fitted_19:
    dictionary of 2019 situation
    :return: plot with hist and boxplot
    '''
    all_depths_09 = []
    hi_depths_09 = []

    for edge, inner_dic in transect_dict_orig_fitted_09.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_depths_09.append(trans_info[6])
                if trans_info[7] > 0.8:
                    hi_depths_09.append(trans_info[6])

    all_depths_19 = []
    hi_depths_19 = []

    for edge, inner_dic in transect_dict_orig_fitted_19.items():
        for skel_pix, trans_info in inner_dic.items():
            # print(trans_info)
            if -30 < trans_info[5] < 30:
                all_depths_19.append(trans_info[6])
                if trans_info[7] > 0.8:
                    hi_depths_19.append(trans_info[6])

    # do the plotting
    boxplotprops_09 = {'patch_artist': True,
                       'boxprops': dict(facecolor='salmon'),
                       'flierprops': dict(marker='o', markerfacecolor='salmon', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='salmon')}
    boxplotprops_19 = {'patch_artist': True,
                       'boxprops': dict(facecolor='teal'),
                       'flierprops': dict(marker='o', markerfacecolor='teal', markersize=0.5, linestyle='none'),
                       'medianprops': dict(color='teal')}
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(6, 5), dpi=300,
                             gridspec_kw={'wspace': 0, 'hspace': 0, 'height_ratios': [2, 2, 6]})
    # fig.tight_layout()
    # axes[0].axis('off')

    median_09, q1_09, q3_09 = np.percentile(hi_depths_09, 50), np.percentile(hi_depths_09, 25), np.percentile(
        hi_depths_09, 75)
    median_19, q1_19, q3_19 = np.percentile(hi_depths_19, 50), np.percentile(hi_depths_19, 25), np.percentile(
        hi_depths_19, 75)

    # 2009 boxplot
    axes[0].boxplot(hi_depths_09, 1, vert=False, widths=0.5, **boxplotprops_09)
    axes[0].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    # axes[0].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    axes[0].set_yticks([])
    axes[0].set_yticklabels([])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].spines['bottom'].set_visible(False)
    axes[0].spines['left'].set_visible(False)
    axes[0].set_ylabel('2009', weight='bold')

    # 2019 boxplot
    axes[1].boxplot(hi_depths_19, 1, vert=False, widths=0.5, **boxplotprops_19)
    axes[1].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9)
    # axes[1].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9)
    axes[1].set_yticks([])
    axes[1].set_yticklabels([])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].set_ylabel('2019', weight='bold')

    # histogram
    # 2009
    axes[2].hist(all_depths_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='peachpuff',
                 label=r"all transects")
    axes[2].hist(hi_depths_09, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='salmon',
                 label=r"transects w/ $r^2 > 0.8$")
    axes[2].axvline(median_09, linestyle='--', color='salmon', alpha=.9, linewidth=.9,
                    label="median")
    # 2019
    axes[2].hist(all_depths_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='powderblue',
                 label=r"all transects")
    axes[2].hist(hi_depths_19, bins=np.arange(0.0, 1.0, 0.02), range=(0, 15), histtype='step', color='teal',
                 label=r"transects w/ $r^2 > 0.8$")
    axes[2].axvline(median_19, linestyle='--', color='teal', alpha=.9, linewidth=.9,
                    label="median")

    axes[2].set_ylabel('frequency')
    axes[2].set_xlabel('depth [m]')
    # axes[0].set_title("Trough Widths")

    # prepare legend
    handles, labels = axes[2].get_legend_handles_labels()
    # colors = ['peachpuff', 'salmon', 'salmon', 'powderblue', 'teal', 'teal']
    # lstyles = ['-', '-', '--', '-', '-', '--']
    # item_melting = mlines.Line2D([], [], color=colors, linestyle=lstyles, linewidth=1)
    # handles[0] = item_melting
    order = [2, 3, 0, 4, 5, 1]
    plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order], loc='upper center',
               bbox_to_anchor=(0.775, 0.875), ncol=1, frameon=False, fontsize=9)
    plt.gcf().text(0.56, 0.416, r'2009', fontsize=10, weight='bold', rotation=90)
    plt.gcf().text(0.56, 0.305, r'2019', fontsize=10, weight='bold', rotation=90)
    # axes[0].subplots_adjust(top=0.5)
    # plt.show()
    plt.savefig('./figures/legend.png')


def do_analysis(transectFile, year, fit_gaussian=True):
    # 2019
    if fit_gaussian:
        transect_dict = load_obj(transectFile)
        transect_dict_fitted = fit_gaussian_parallel(transect_dict)
        save_obj(transect_dict_fitted, 'arf_transect_dict_fitted_' + year)

    transect_dict_fitted = load_obj('arf_transect_dict_fitted_' + year + '.pkl')
    edge_param_dict = get_trough_avgs_gauss(transect_dict_fitted)
    save_obj(edge_param_dict, 'arf_transect_dict_avg_' + year)

    return transect_dict_fitted, edge_param_dict


if __name__ == '__main__':
    pkl = sys.argv[1]
    year = sys.argv[1].split(".")[0].split("_")[3]
    print(year)

    # pkl = 'E:/02_macs_fire_sites/00_working/03_code_scripts/IWD_graph_analysis/data/graphs/arf_transect_dict_2009.pkl'

    transect_dict_fitted, edge_param_dict = do_analysis(pkl, year, True)

    print(datetime.now() - startTime)

    # plt.show()

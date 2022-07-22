# Import necessary modules
import pandas as pd
import geopandas as gpd
from osgeo import gdal
import ogr
from datetime import datetime
import matplotlib
from osgeo import osr, gdal_array
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage.morphology import skeletonize, medial_axis, skeletonize_3d
import sknw
import glob
import csv
from affine import Affine
# import shapefile as shp

# change the global options that Geopandas inherits from
pd.set_option('display.max_columns', None)

startTime = datetime.now()

plt.Figure()

path_footprints = 'E:/02_macs_fire_sites/00_working/00_orig-data/03_lidar/lidar_footprints_merged.shp'
path_aoi = 'E:/02_macs_fire_sites/00_working/01_processed-data/00_study_area_shp/sa_025_firescars'
path_clipped = 'E:/02_macs_fire_sites/00_working/00_orig-data/03_lidar/'


def set_epsg(dataset, epsg_code):
    dataset = dataset.to_crs(epsg_code)
    return dataset


def clip_vec_to_aoi(path_clip_feature, path_aoi, path_clipped):
    # read in aoi shapefile
    aoi = gpd.read_file(path_aoi)
    aoi_fn = path_aoi[86:]
    print(aoi_fn)
    # aoi_epsg = aoi.crs
    aoi = set_epsg(aoi, 4326)

    # Read in shapefile of lidar footprints based on name
    footprints = gpd.read_file(path_clip_feature)
    footprints = set_epsg(footprints, 4326)

    # buffer(0) needs to be computed to avoid a TopologicalError bug
    # see also: https://stackoverflow.com/questions/63955752/topologicalerror-the-operation-geosintersection-r-could-not-be-performed
    footprints['geometry'] = footprints.buffer(0)

    footprints_aoi = gpd.clip(footprints, aoi)
    fn = footprints_aoi['path'].str[-48:]
    # print(fn)
    footprints_aoi['footprint_name'] = fn

    footprints_aoi.to_file(path_clipped + 'lidar_fprts_clip2aoi/' + aoi_fn)

    name = aoi_fn
    epsg = footprints_aoi.crs
    geom = str(aoi['geometry'][0])[10:-2]
    lidar_fprts = fn.to_list()
    #
    with open(path_clipped + 'csv/' + aoi_fn[:-4] + '.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow([name, epsg, geom, lidar_fprts])

    return "footprints_aoi"


def merge_csvs(directory):
    filenames = os.listdir(directory + 'csv/')
    fout = open(f'{directory}merged_csv_temp.csv', "a")
    for file in filenames:
        print(file)
        f = open(f'{directory}csv/{file}')
        for line in f:
            fout.write(line)
        f.close()
    fout.close()

    # now remove the blank lines... --> figure out easier way... also, why are there blank lines in the first place...
    df = pd.read_csv(f'{directory}merged_csv_temp.csv')
    df.to_csv(f'{directory}merged_csv.csv', index=False)


if __name__ == '__main__':
    # for filename in glob.iglob(f'{path_aoi}/*.shp'):
    #     print(filename)
    #     clip_vec_to_aoi(path_footprints, filename, path_clipped)

    merge_csvs(path_clipped)

    # print time needed for script execution
    print(datetime.now() - startTime)

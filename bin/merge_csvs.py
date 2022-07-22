#!/usr/bin/env python

import sys
import os
import glob
import pandas as pd

if __name__ == '__main__':

    years = sys.argv[1:]

    print(years)

    path = './'
    extension = 'csv'
    os.chdir(path)
    listFiles = glob.glob('*.{}'.format(extension))

    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

    combined_csv.to_csv( "merged_csv.csv", index=False, encoding='utf-8-sig')

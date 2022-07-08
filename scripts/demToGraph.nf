#!/usr/bin/env python


nextflow.enable.dsl=2

process demToGraph {

    container 'fondahub/iwd:latest'

    input:
        path yearFile
    output:
    path "*.tif", emit: tif
    path "*.npy", emit: npy
    path "*.edgelist", emit: edgelist

    script:
    """
    a_dem_to_graph.py ${yearFile}
    """

}
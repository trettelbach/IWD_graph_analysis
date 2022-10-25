#!/usr/bin/env python


nextflow.enable.dsl=2

process demToGraph {
    publishDir 'output/tifs', mode: 'copy', pattern: '*.tif'
    container 'fondahub/iwd:latest'

    input:
        tuple val(key), file(yearFile)
        val(version)

    output:
    tuple val(key), path("*skel.tif"), path("*.npy"), path("*.edgelist"), emit:tup
    path("*_?.tif")

    script:
    """
    a_dem_to_graph.py ${yearFile} ${version}
    """

}
#!/usr/bin/env python


nextflow.enable.dsl=2

process demToGraph {

    container 'fondahub/iwd:latest'

    input:
        tuple val(key), file(yearFile)
        val(version)

    output:
    tuple val(key), path("*.tif"), path("*.npy"), path("*.edgelist")

    script:
    """
    a_dem_to_graph.py ${yearFile} ${version}
    """

}
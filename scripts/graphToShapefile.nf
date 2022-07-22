nextflow.enable.dsl=2

process graphToShapefile {

    container 'fondahub/iwd:latest'

    input:
        path edgelist
        path npy
        path weighted_graph_edgelist


    output:
        path "*nodes.shp", emit: nodes
        path "*edges.shp", emit: edges

    script:
    """
    e_graph_to_shapefile.py ${edgelist} ${npy} ${weighted_graph_edgelist}
    """

}
nextflow.enable.dsl=2

process graphToShapefile {

    container 'fondahub/iwd:latest'

    input:
        path edgelist
        path npy
        path shp_loc
        path weighted_graph_edgelist
        path csv_loc


    output:
        path "*nodes.shp", emit: nodes
        path "*edges.shp", emit: edges

    script:
    """
    e_graph_to_shapefile.py ${edgelist} ${npy} ${shp_loc} ${weighted_graph_edgelist} ${csv_loc}
    """

}
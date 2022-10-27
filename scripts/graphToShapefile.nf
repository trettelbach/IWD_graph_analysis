nextflow.enable.dsl=2

process graphToShapefile {
    publishDir 'output/shp', mode: 'copy', pattern: '*_edges.*'
    publishDir 'output/shp', mode: 'copy', pattern: '*_nodes.*'
    container 'fondahub/iwd:latest'

    input:
        tuple val(key), path(npy), path(edgelist), path(weighted_graph_edgelist)


    output:
        path "*nodes.*", emit: nodes
        path "*edges.*", emit: edges

    script:
    """
    e_graph_to_shapefile.py ${edgelist} ${npy} ${weighted_graph_edgelist}
    """

}
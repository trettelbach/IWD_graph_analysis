nextflow.enable.dsl=2

process networkAnalysis {
    publishDir 'output/csv', mode: 'copy', pattern: '**.csv'
    container 'fondahub/iwd:latest'

    input:
        path edgelist
        path npy
        path transect_dict_avg


    output:
        path("graph_????.csv"), emit: csv

    script:
    """
    d_network_analysis.py ${edgelist} ${npy} ${transect_dict_avg}
    """

}
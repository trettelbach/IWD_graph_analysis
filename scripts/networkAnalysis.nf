nextflow.enable.dsl=2

process networkAnalysis {

    container 'was1docker/iwd:latest'

    input:
        path edgelist
        path npy
        path transect_dict_avg


    output:


    script:
    """
    d_network_analysis.py ${edgelist} ${npy} ${transect_dict_avg}
    """

}
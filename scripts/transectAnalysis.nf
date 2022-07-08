nextflow.enable.dsl=2

process transectAnalysis {

    container 'was1docker/iwd:latest'

    input:
        path pkl

    output:
    path "*transect_dict_avg*", emit: transect_dict_avg


    script:
    """
    c_transect_analysis.py ${pkl}
    """

}
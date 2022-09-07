nextflow.enable.dsl=2

process transectAnalysis {

    container 'fondahub/iwd:latest'

    input:
        tuple val(key), path(pkl)
        val(version)

    output:
        tuple val(key), path("*transect_dict_avg*")


    script:
    """
    c_transect_analysis.py ${pkl} ${version}
    """

}
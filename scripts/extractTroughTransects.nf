nextflow.enable.dsl=2

process extractTroughTransects {

    container 'fondahub/iwd:latest'

    input:
        tuple val(key), file(tif), file(npy), file(edgelist)
        val(version)

    output:
        tuple val(key), path("*.pkl")

    script:
    """
    b_extract_trough_transects.py ${tif} ${npy} ${edgelist} ${version}
    """

}
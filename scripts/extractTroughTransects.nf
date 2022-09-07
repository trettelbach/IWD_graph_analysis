nextflow.enable.dsl=2

process extractTroughTransects {

    container 'fondahub/iwd:latest'

    input:
        tuple val(key), file(npy), file(edgelist), file(tif)
        val(version)

    output:
        tuple val(key), path("*.pkl")

    script:
    """
    b_extract_trough_transects.py ${edgelist} ${npy} ${tif} ${version}
    """

}
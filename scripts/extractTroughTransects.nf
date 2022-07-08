nextflow.enable.dsl=2

process extractTroughTransects {

    container 'fondahub/iwd:latest'

    input:
        path npy
        path edgelist
        path tif


    output:
        path "*.pkl", emit: pkl

    script:
    """
    b_extract_trough_transects.py ${edgelist} ${npy} ${tif}
    """

}
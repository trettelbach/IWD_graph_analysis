nextflow.enable.dsl=2

process networkAnalysis {
    publishDir 'output/csv', mode: 'copy', pattern: '**.csv'
    container 'fondahub/iwd:latest'

    input:
        tuple val(key), path(tif), path(npy), path(edgelist), path(transect_dict_avg)
        val(version)

    output:
        tuple val(key), path("graph_*.csv"), emit: csvs
        tuple val(key), path("*weights.edgelist"), emit: weightedEdgelist

    script:
    """
    d_network_analysis.py ${tif} ${edgelist} ${npy} ${transect_dict_avg} ${version}
    """

}
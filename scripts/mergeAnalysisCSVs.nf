nextflow.enable.dsl=2

process mergeAnalysisCSVs {
    publishDir 'output/csv', mode: 'copy'
    container 'fondahub/iwd:latest'

    input:
        path("*")

    output:
    path "merged_csv.csv"


    script:
    """
    merge_csvs.py
    """

}
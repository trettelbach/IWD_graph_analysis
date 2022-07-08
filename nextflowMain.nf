nextflow.enable.dsl=2

include { demToGraph } from './scripts/demToGraph'
include { extractTroughTransects } from './scripts/extractTroughTransects'
include { transectAnalysis } from './scripts/transectAnalysis'
include { networkAnalysis } from './scripts/networkAnalysis'



//Main workflow
workflow {

    data = Channel.fromPath( 'data/*dtm*.tif' ).view { "value: $it" }



    //py_path3 = Channel.fromPath('${params.maindir}/interpol_net_goce_simple.py')
    //py_path4 = Channel.fromPath('${params.maindir}/interpol_net_goce_simple_finetune.py')
    //py_path5 = Channel.fromPath('${params.maindir}/publication_write_cdf_files_goce_simple.py')

    demToGraph(data)
    extractTroughTransects(demToGraph.out)
    transectAnalysis(extractTroughTransects.out)
    networkAnalysis(demToGraph.out.edgelist, demToGraph.out.npy, transectAnalysis.out)

    //preprocessing2.out.view()
    //interpol(preprocessing2.out(),py_path3)
    //interpol_finetune(interpol.out(),py_path4)
    //publication(interpol_finetune.out(),py_path5)
}

nextflow.enable.dsl=2

include { demToGraph } from './scripts/demToGraph'
include { extractTroughTransects } from './scripts/extractTroughTransects'
include { transectAnalysis } from './scripts/transectAnalysis'
include { networkAnalysis } from './scripts/networkAnalysis'
include { mergeAnalysisCSVs} from './scripts/mergeAnalysisCSVs'
include { graphToShapefile } from './scripts/graphToShapefile'




//Main workflow
workflow {

    version = 2

    data = null

    print(version == 2)

    if (version == 2) {
        data = Channel.fromPath( 'data/v'+ version +'/PERMAX*.tif' ).map { file ->
           return new Tuple(file.getName().split("\\.")[0].substring(13), file)
        }.view { "value: $it" }
    } else if (version == 1) {
        data = Channel.fromPath( 'data/v'+ version +'/*dtm*.tif' ).map { file ->
        print(file.getName())
               return new Tuple(file.getName().split("\\.")[0].substring(8), file)
            }

    }




    //py_path3 = Channel.fromPath('${params.maindir}/interpol_net_goce_simple.py')
    //py_path4 = Channel.fromPath('${params.maindir}/interpol_net_goce_simple_finetune.py')
    //py_path5 = Channel.fromPath('${params.maindir}/publication_write_cdf_files_goce_simple.py')

    demToGraph(data, version)
    extractTroughTransects(demToGraph.out, version)
    transectAnalysis(extractTroughTransects.out, version)

    networkAnalysisInput = demToGraph.out.join(transectAnalysis.out)

    networkAnalysis(networkAnalysisInput, version)
    // TODO hier muss match über Name erfolgen
    //graphToShapefile(demToGraph.out.edgelist, demToGraph.out.npy, transectAnalysis.out)

    csv = networkAnalysis.out.map{it[1]}flatten().buffer( size: Integer.MAX_VALUE, remainder: true )

    mergeAnalysisCSVs(csv)



}

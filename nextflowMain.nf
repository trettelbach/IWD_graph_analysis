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

    if(params.version && params.version <= 2 && params.version >= 1) {
        version = params.version
    }




    data = null

    if (version == 2) {
        data = Channel.fromPath( 'data/v'+ version +'/PERMAX*.tif' ).map { file ->
           return new Tuple(file.getName().split("\\.")[0].substring(13), file)
        }
    } else if (version == 1) {
        data = Channel.fromPath( 'data/v'+ version +'/*dtm*.tif' ).map { file ->
        print(file.getName())
               return new Tuple(file.getName().split("\\.")[0].substring(8), file)
            }

    }

    demToGraph(data, version)
    extractTroughTransects(demToGraph.out.tup, version)
    transectAnalysis(extractTroughTransects.out, version)

    networkAnalysisInput = demToGraph.out.tup.join(transectAnalysis.out)

    networkAnalysis(networkAnalysisInput, version)

    graphToShapefileInput = demToGraph.out.tup.join(transectAnalysis.out)
    graphToShapefileInput = graphToShapefileInput.join(networkAnalysis.out.weightedEdgelist)
    graphToShapefileInput = graphToShapefileInput.map{it -> it[0,2,3,5]}

    graphToShapefile(graphToShapefileInput)

    csv = networkAnalysis.out.csvs.map{it[1]}flatten().buffer( size: Integer.MAX_VALUE, remainder: true )

    mergeAnalysisCSVs(csv)

}

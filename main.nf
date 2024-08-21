// enabling nextflow DSL v2
nextflow.enable.dsl=2

// defaults results directory
resultsDir = params.resultsDir ? params.resultsDir : "${workflow.launchDir}/results"

// expDir = "${resultsDir}/${params.geneFile}/${params.networkFile}"

println ""
println "Gene file: ${params.geneFile}"
println "Network file: ${params.networkFile}"
println "Results dir: ${resultsDir}"
println "Training set size: ${params.train_size}"
println "Models: ${params.models}"
println "Learning rate: ${params.learning_rate}"
println "Weight decay: ${params.weight_decay}"
println "Epochs: ${params.epochs}"
println "Replicates: ${params.replicates}"
println "Metrics:  ${params.metrics}"
println "Eval-q: ${params.eval_threshold}"
println "Verbose interval: ${params.verbose_interval}"

println ""


dataSet = params.dataSet

process TrainGNN {
    tag "${model}-${epoch}"
    
    publishDir "${resultsDir}/data/${dataSet}", pattern: "full-${model}-${epoch}-run-${run}*.txt", mode: 'copy'

    input:
        tuple path(geneFile), path(networkFile), val(model), val(epoch)
        each run 

    output:
        tuple val(model), val(epoch), path("full-${model}-${epoch}-run-${run}.txt"), emit: full_output
        tuple val(model), val(epoch), val("train"), path("full-${model}-${epoch}-run-${run}-train.txt"), emit: train_output
        tuple val(model), val(epoch), val("test"), path("full-${model}-${epoch}-run-${run}-test.txt"), emit: test_output
        tuple val(model), val(epoch), val("all"), path("full-${model}-${epoch}-run-${run}-all.txt"), emit: all_output

    """
        gnn.py ${geneFile} ${networkFile} \
                --train-size ${params.train_size} \
                --model-name ${model} \
                --learning-rate ${params.learning_rate} \
                --weight-decay ${params.weight_decay} \
                --epochs ${epoch} \
                --eval-threshold ${params.eval_threshold} \
                --verbose-interval ${params.verbose_interval} \
                --dropout ${params.dropout} \
                --alpha ${params.alpha} \
                --theta ${params.theta} \
                 > full-${model}-${epoch}-run-${run}.txt

        split_data.py full-${model}-${epoch}-run-${run}.txt
    """
}



process PlotEpochMetrics {
    tag "${model}-${epoch}-${split}"

    publishDir "${resultsDir}/figures/${dataSet}", pattern: "${model}-${epoch}-split-${split}*.pdf", mode: 'copy'

    input:
        tuple val(model), val(epoch), val(split), path(files)
       

    output:
        path "${model}-${epoch}-split-${split}.pdf"

    script:
    """
        plot.py --model ${model} ${model}-${epoch}-split-${split}.pdf ${files}
    """
}


process ComputeStats {
    tag "${model}"

    input:
        tuple val(model), path(results,stageAs: "?/*")

    output:
        path "stats-${model}.txt"

    """
        stats.py compute stats-${model}.txt ${results} ${model}
    """
}

process CollectStats {
    tag "Final stats"

    publishDir "${resultsDir}", pattern: "stats.tex", mode: 'copy'

    input:
        path results

    output:
        path "stats.tex"

    """
        # collect stats
        stats.py collect stats.tex ${results}
    """
}

process HyperparameterOptimization {
    input:
    path config_file
    path data_files
  

    output:
    path "optuna_${params.dataSet}.json" into optuna_results

    script:
    """
    python3 hyperopt_all_2.py --config ${config_file} --data ${data_files} --output optuna_${params.dataSet}.json
    """
}

workflow hyperoptWorkflow {
    HyperparameterOptimization(config_file: params.config, data_files: params.data)
}


workflow {
    // building channels for experiments
    geneChan = channel.fromPath(params.geneFile)
    networkChan = channel.fromPath(params.networkFile)
    modelChan = channel.from(params.models)
    epochChan = channel.from(params.epochs)
    replicatesChan = channel.of(1..params.replicates)
    metricsChan = channel.from(params.metrics)
    
    // benchmark channel
    benchmarkChan = geneChan.combine(networkChan).combine(modelChan).combine(epochChan) 
    
    // train the GNN
    results = TrainGNN(benchmarkChan, replicatesChan)
    //results.train_output.view()

    // group the splitDataResults by model and epoch
    groupedResult_train = results.train_output
        .groupTuple(by: [0,1, 2]) // group by model and epoch: [model, epoch, list(runs), list(train-files), list(test-files)]
    groupedResult_test = results.test_output
        .groupTuple(by: [0,1, 2]) 
    groupedResult_all = results.all_output
        .groupTuple(by: [0,1, 2]) 
    
    groupedResult = groupedResult_train.concat(groupedResult_test, groupedResult_all)
    groupedResult.view()

    // plot the results
    PlotEpochMetrics(groupedResult).view()
}


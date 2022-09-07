# IWD_graph_analysis
A repository containing data and scripts for the publication 

**A Quantitative Graph-Based Approach to Monitoring Ice-Wedge Trough Dynamics in Polygonal Permafrost Landscapes.**
Rettelbach, T.; Langer, M.; Nitze, I.; Jones, B.; Helm, V.; Freytag, J.-C.; Grosse, G.
_Remote Sens._ 2021, 13, 3098. https://doi.org/10.3390/rs13163098

![graphical_abstract_IWD_analysis](https://user-images.githubusercontent.com/40014163/128493596-35c15cca-0405-4c83-9ea8-c23401cf83c3.png)


## Run as Nextflow Workflow

Before you start, set the version number with --version 1|2

`nextflow run nextflowMain.nf -with-docker fondahub/iwd:latest -with-report --version 2`
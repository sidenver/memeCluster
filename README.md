Meme Cluster
===

Here, we define the steps used for meme clustering. 

Step 1: Preprocessing
Download data from http://snap.stanford.edu/memetracker/srcdata/clust-qt08080902w3mfq5.txt.gz
Run util.py in the same repository. It will generate a phrases.gz file which contains all phrases. 

Step 2: First step clustering
Run kmeans_clustering.py in the same repository to generate 1000 clusters. It will write the clusters to clusters.json file. 

Step 3: 



Pre-clustering
input - raw date
output - json {bucket_id: [list of phrases]}

Algorithm
input - json {bucket_id: [list of phrases]}
output - json {cluster_id: [list of phrases]}

Evaluation
input - json {cluster_id: [list of phrases]}
output - score


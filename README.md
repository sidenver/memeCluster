Meme Cluster
===

Here, we define the steps used for meme clustering. 

*Step 1: Preprocessing*

Download data from http://snap.stanford.edu/memetracker/srcdata/clust-qt08080902w3mfq5.txt.gz
Run util.py in the same repository. It will generate a phrases.gz file which contains all phrases. 

*Step 2: First step clustering*

Run kmeans_clustering.py in the same repository to generate 1000 buckets. It will write the buckets to clusters.json file. 

*Step 3: Compute Distance Matrix*

Run distanceW2V.py to calculate NeedleWunsch distance with word2vec and generate distance matrixes for all buckets and save them as csv files.

*Step 4: Run Hierachical Clustering Algorithm and Evaluate*

Run clustering.py to do hierachical clustering on each bucket based on the corresponding distance matrix. This step will also invoke evaluateCluster.py to generate clustering performance scores.

*Step 5: Plot Result*

Run testPlot.py to plot the result.





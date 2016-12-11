
# Packages Required : JSON, NUMPY, SCIPY
import pickle
import simplejson as json
import numpy
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete, average
from collections import defaultdict
import os
import evaluateCluster
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib

# Location of the json file
filename = 'updated_clusters.json'

# Dictionary containing all the buckets
buckets = json.loads(open(filename).read())


def diffCluster(matDist, threshold, labels, clusteringType):
    if clusteringType == 1:
        linkage_matrix = ward(matDist)
    elif clusteringType == 2:
        linkage_matrix = single(matDist)
    elif clusteringType == 3:
        linkage_matrix = complete(matDist)
    elif clusteringType == 4:
        linkage_matrix = average(matDist)
    else:
        return {}
    cluster_labels = fcluster(linkage_matrix, threshold)
    clusters_dict = defaultdict(list)
    for sent, cluster_id in zip(cluster_labels, labels):
        clusters_dict[cluster_id].append(sent)
    return clusters_dict

# Global Alignment method

distMat = {}  # Dictionary of Distance Matrices. distMat [bucketID] => Distance Matrix
bucketPhrases = {}  # Dictionary of Phrases. phraseList [bucketID] => Phrase List in the bucket


totalBuckNum = str(len(buckets))

path = "/fs/clip-scratch/shing/memeW2V/"

for filename in os.listdir(path):
    if filename.split('.')[-1] == 'csv':
        key = filename.split('.')[0]
        distMat[key] = numpy.loadtxt(open(path + filename, "rb"), delimiter=",")
        elem = buckets[key]
        bucketPhrases[key] = []
        for i in range(0, len(elem)):
            bucketPhrases[key].append(elem[i])
print 'loaded {} csv files'.format(str(len(distMat)))

gold = evaluateCluster.loadAndFilterGold('raw_phrases', bucketPhrases.values())
print 'loaded gold'

if __name__ == '__main__':
    # Clustering starts here ....
    wardAssignment = {}  # clusterAssignment [bucketID] => {dict[phrase] => ClusterID }
    singleAssignment = {}  # nearest neighbor
    completeAssignment = {}  # farthest neighbor
    averageAssignment = {}  # Average distance
    clusterAssignments = {'ward': wardAssignment,
                          'single': singleAssignment,
                          'complete': completeAssignment,
                          'average': averageAssignment}
    wardScore = {}
    singleScore = {}
    completeScore = {}
    averageScore = {}
    clusterScore = {'ward': wardScore,
                    'single': singleScore,
                    'complete': completeScore,
                    'average': averageScore}

    for threshold in [0.0, 0.025, 0.05, 0.1, 0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for key in bucketPhrases:
            wardAssignment[key] = diffCluster(distMat[key], threshold, bucketPhrases[key], 1)
            singleAssignment[key] = diffCluster(distMat[key], threshold, bucketPhrases[key], 2)
            completeAssignment[key] = diffCluster(distMat[key], threshold, bucketPhrases[key], 3)
            averageAssignment[key] = diffCluster(distMat[key], threshold, bucketPhrases[key], 4)

        for clusterMethod in clusterAssignments:
            clusterScore[clusterMethod][threshold] = {}
            answer = evaluateCluster.makeId2sentenceList(clusterAssignments[clusterMethod])
            print '{} with threshold {}'.format(clusterMethod, str(threshold))
            clusterEvaluator = evaluateCluster.ClusterEvaluator(gold, answer)
            clusterScore[clusterMethod][threshold]['Purity'] = clusterEvaluator.calPurity()
            clusterScore[clusterMethod][threshold]['NMI'] = clusterEvaluator.calNMI()
            clusterScore[clusterMethod][threshold]['Adjust_RI'] = clusterEvaluator.calRI()

    pickle.dump(clusterScore, open('plotResultW2V.out', "wb"))

    # matplotlib.style.use('ggplot')
    # for method in clusterScore:
    #     df = pd.DataFrame(clusterScore[method])
    #     df = df.transpose()
    #     plt.figure()
    #     df.plot()
    #     plt.savefig('{}.png'.format(method))

    # answer=pickle.load(open('output.out',"rb"))
    # print clusterAssignment

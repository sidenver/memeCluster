
# Packages Required : JSON, NUMPY, SCIPY
import pickle
import simplejson as json
import numpy
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete, average
from collections import defaultdict
import os
import evaluateCluster

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

path = "/fs/clip-scratch/shing/meme/"

for filename in os.listdir(path):
    if filename.split('.')[-1] == 'csv':
        key = filename.split('.')[0]
        distMat[key] = numpy.loadtxt(open(path + filename, "rb"), delimiter=",")
        elem = buckets[key]
        bucketPhrases[key] = []
        for i in range(0, len(elem)):
            bucketPhrases[key].append(elem[i])

print 'loaded {} csv files'.format(str(len(distMat)))

if __name__ == '__main__':
    # Clustering starts here ....
    wardAssignment = {}  # clusterAssignment [bucketID] => {dict[phrase] => ClusterID }
    singleAssignment = {}  # nearest neighbor
    completeAssignment = {}  # farthest neighbor
    averageAssignment = {}  # Average distance

    for key in bucketPhrases:
        wardAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 1)
        singleAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 2)
        completeAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 3)
        averageAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 4)

    answer = evaluateCluster.makeId2sentenceList(wardAssignment)
    print 'loaded answer ward'
    gold = evaluateCluster.loadAndFilterGold('raw_phrases', answer)
    print 'loaded gold'
    clusterEvaluator = evaluateCluster.ClusterEvaluator(gold, answer)
    print 'purity', clusterEvaluator.calPurity()
    print 'NMI', clusterEvaluator.calNMI()
    print 'Adjust RI', clusterEvaluator.calRI()
    # answer=pickle.load(open('output.out',"rb"))
    # print clusterAssignment

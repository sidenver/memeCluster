
# Packages Required : JSON, NUMPY, SCIPY
import pickle
import simplejson as json
import numpy
from scipy.cluster.hierarchy import ward, dendrogram, fcluster, single, complete, average
from collections import defaultdict
import os

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


def NeedleWunsch(str1, str2):
    L1 = str1.split()
    L2 = str2.split()
    len1 = len(L1)
    len2 = len(L2)
    dp = numpy.zeros((len1 + 1, len2 + 1), int)
    for x in range(0, len1):
        dp[x][0] = x
    for x in range(0, len2):
        dp[0][x] = x

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = min(dp[i - 1][j - 1] + (L1[i - 1] != L2[j - 1]), dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    return dp[len1][len2]

# Local Alignment method


def SmithWaterman(str1, str2):
    L1 = str1.split()
    L2 = str2.split()
    len1 = len(L1)
    len2 = len(L2)
    dp = numpy.zeros((len1 + 1, len2 + 1), int)

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dp[i][j] = max(0, dp[i - 1][j - 1] + (L1[i - 1] != L2[j - 1]), dp[i - 1][j] + 1, dp[i][j - 1] + 1)

    return dp[len1][len2]

distMat = {}  # Dictionary of Distance Matrices. distMat [bucketID] => Distance Matrix
bucketPhrases = {}  # Dictionary of Phrases. phraseList [bucketID] => Phrase List in the bucket


totalBuckNum = str(len(buckets))

for filename in os.listdir("/fs/clip-scratch/shing/meme/"):
    if filename.split('.')[-1] == 'csv':
        key = filename.split('.')[0]
        distMat[key] = numpy.loadtxt(open(filename, "rb"), delimiter=",")
        elem = buckets[key]
        bucketPhrases[key] = []
        for i in range(0, len(elem)):
            bucketPhrases[key].append(elem[i])

# for idx, key in enumerate(buckets):
#     print 'processing bucket: {}/{}'.format(str(idx), totalBuckNum)
#     # Some threshold can be added here
#     elem = buckets[key]
#     bucketPhrases[key] = []
#     distMat[key] = numpy.zeros((len(elem), len(elem)), int)
#     for i in range(0, len(elem)):
#         bucketPhrases[key].append(elem[i])
#         for j in range(i + 1, len(elem)):
#             distMat[key][i][j] = distMat[key][j][i] = NeedleWunsch(elem[i], elem[j])
#             # print i + 1, '-', j + 1, '  : ', distMat[key][j][i]
#     numpy.savetxt("/fs/clip-scratch/shing/meme/{}.csv".format(key), distMat[key], delimiter=",")


# Clustering starts here ....
wardAssignment = {}  # clusterAssignment [bucketID] => {dict[phrase] => ClusterID }
singleAssignment = {}  # nearest neighbor
completeAssignment = {}  # farthest neighbor
averageAssignment = {}  # Average distance

for key in bucketPhrases:
    wardAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 1)
pickle.dump(wardAssignment, open('wardOutput.out', "wb"))

for key in bucketPhrases:
    singleAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 2)
pickle.dump(singleAssignment, open('singleOutput.out', "wb"))

for key in bucketPhrases:
    completeAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 3)
pickle.dump(completeAssignment, open('completeOutput.out', "wb"))

for key in bucketPhrases:
    averageAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key], 4)
pickle.dump(averageAssignment, open('averageOutput.out', "wb"))
# answer=pickle.load(open('output.out',"rb"))
# print clusterAssignment

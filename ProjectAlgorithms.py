# Packages Required : JSON, NUMPY, SCIPY
import pickle
import simplejson as json
import numpy
from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from collections import defaultdict


# Location of the json file
filename = 'updated_clusters.json'

# Dictionary containing all the buckets
buckets = json.loads(open(filename).read())


def diffCluster(matDist, threshold, labels):
    linkage_matrix = ward(matDist)
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
clusterAssignment = {}  # clusterAssignment [bucketID] => {dict[phrase] => ClusterID }

totalBuckNum = str(len(buckets))

for idx, key in enumerate(buckets):
    print 'processing bucket: {}/{}'.format(str(idx), totalBuckNum)
    # Some threshold can be added here
    elem = buckets[key]
    bucketPhrases[key] = []
    distMat[key] = numpy.zeros((len(elem), len(elem)), int)
    for i in range(0, len(elem)):
        bucketPhrases[key].append(elem[i])
        for j in range(i + 1, len(elem)):
            distMat[key][i][j] = distMat[key][j][i] = NeedleWunsch(elem[i], elem[j])
            # print i + 1, '-', j + 1, '  : ', distMat[key][j][i]

for key in buckets:
    clusterAssignment[key] = diffCluster(distMat[key], 0.5, bucketPhrases[key])

pickle.dump(clusterAssignment, open('output.out', "wb"))
# answer=pickle.load(open('output.out',"rb"))
# print clusterAssignment

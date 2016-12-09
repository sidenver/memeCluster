from collections import Counter, defaultdict
import math
import operator as op
import pickle


class ClusterEvaluator(object):
    def __init__(self, gold, result):
        # gold is list of list of sentences
        # result is list of list of sentences
        self.gold = gold
        self.result = result
        self.totalNum = float(sum([len(cluster) for cluster in self.gold]))
        print 'gold count: ', self.totalNum
        print 'result count: ', float(sum([len(cluster) for cluster in self.result]))
        self.sentence2GoldId = {}
        self.buildSentence2GoldId()
        self.indxListList = self.cluster2Indx()

    def buildSentence2GoldId(self):
        for idx, cluster in enumerate(self.gold):
            for sentence in cluster:
                self.sentence2GoldId[sentence] = idx

    def getSentenceIdinGold(self, sentence):
        return self.sentence2GoldId[sentence]

    def cluster2Indx(self):
        indxListList = []
        for cluster in self.result:
            indxList = []
            for sentence in cluster:
                goldId = self.getSentenceIdinGold(sentence)
                indxList.append(goldId)
            indxListList.append(indxList)
        return indxListList

    def calMutualInfo(self):
        mutualInfo = 0.0
        for clusterResultInId in self.indxListList:
            idCounter = Counter(clusterResultInId)
            resultCount = float(len(clusterResultInId))
            for goldId, clusterGold in enumerate(self.gold):
                mutualCount = idCounter[goldId]
                goldCount = float(len(clusterGold))
                mutualInfo += mutualCount/self.totalNum*math.log(self.totalNum*mutualCount/(resultCount*goldCount))

        return mutualInfo

    def calEntropy(self, clusterList):
        entropy = 0.0
        for cluster in clusterList:
            entropy -= len(cluster)/self.totalNum*math.log(len(cluster)/self.totalNum)
        return entropy

    def calPurity(self):
        """
        High purity is easy to achieve when the number of clusters is large
        - in particular, purity is 1 if each document gets its own cluster.
        Thus, we cannot use purity to trade off the quality of the clustering
        against the number of clusters.
        """
        maxCountList = [Counter(indxList).most_common(1)[0][1] for indxList in self.indxListList]
        return sum(maxCountList)/self.totalNum

    def calNMI(self):
        """
        normalized mutual information or NMI
        """
        mutualInfo = self.calMutualInfo()
        entropyGold = self.calEntropy(self.gold)
        entropyResult = self.calEntropy(self.result)
        return mutualInfo/((entropyGold+entropyResult)/2.0)

    def ncr(self, n, r):
        r = min(r, n-r)
        if r == 0:
            return 1
        numer = reduce(op.mul, xrange(n, n-r, -1))
        denom = reduce(op.mul, xrange(1, r+1))
        return numer//denom

    def calRI(self):
        """
        The Rand index measures the percentage of decisions that are correct.
                  TP+TN
        RI = ---------------
               TP+TN+FP+FN
        """
        tp_fp = sum([self.ncr(len(cluster), 2) for cluster in self.indxListList if len(cluster) >= 2])
        countList = [Counter(indxList) for indxList in self.indxListList]
        tp = sum([sum([self.ncr(count, 2) for idx, count in counter.most_common() if count >= 2]) for counter in countList])
        fp = tp_fp - tp
        tn_fn = 0
        for idx, cluster in enumerate(self.indxListList[:-1]):
            thisClusterCount = len(cluster)
            afterClusterCount = sum([len(afterCluster) for afterCluster in self.indxListList[idx+1:]])
            tn_fn += thisClusterCount*afterClusterCount
        fn = 0
        for idx, counter in enumerate(countList[:-1]):
            for key in counter:
                afterClusterCount = sum([afterCounter[key] for afterCounter in countList[idx+1:]])
                fn += afterClusterCount*counter[key]
        tn = tn_fn - fn
        accuracy = (tp+tn)/float(tp+tn+fp+fn)
        precision = tp/float(tp+fp)
        recall = tp/float(tp+fn)
        return accuracy, precision, recall


def loadAndFilterGold(filename, listofSentenceList):
    sentenceSet = set()
    for sentenceList in listofSentenceList:
        sentenceSet.update(set(sentenceList))
    id2sentenceList = defaultdict(list)
    with open(filename, 'r') as f:
        for idx, line in enumerate(f):
            if not idx == 0:
                field = line.split('\t')
                if field[2] in sentenceSet:
                    id2sentenceList[field[0]].append(field[2])
    listofSentenceList = []
    for newId in id2sentenceList:
        listofSentenceList.append(id2sentenceList[newId])
    return listofSentenceList


def makeId2sentenceList(dictOfdictList):
    id2sentenceList = defaultdict(list)
    for bucketid in dictOfdictList:
        for sentence in dictOfdictList[bucketid]:
            for assigment in dictOfdictList[bucketid][sentence]:
                id2sentenceList[bucketid + '_' + str(assigment)].append(sentence)
    listofSentenceList = []
    for newId in id2sentenceList:
        listofSentenceList.append(id2sentenceList[newId])
    return listofSentenceList

if __name__ == '__main__':
    answer = makeId2sentenceList(pickle.load(open('averageOutput.out', "rb")))
    print 'loaded answer'
    gold = loadAndFilterGold('raw_phrases', answer)
    print 'loaded gold'
    clusterEvaluator = ClusterEvaluator(gold, answer)
    purity = clusterEvaluator.calPurity()
    nmi = clusterEvaluator.calNMI()
    ri = clusterEvaluator.calRI()
    print purity, nmi, ri

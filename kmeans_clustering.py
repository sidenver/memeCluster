from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans, MiniBatchKMeans
from collections import defaultdict
import gzip
import json

def read_phrases(file_name):
    """
    Read phrases from the phrases.gz file and create a python list of phrases. 
    """
    docs = []
    doc_count = 0
    with gzip.open(file_name, 'r') as gz_file:
        for line in gz_file:
            line = line.decode('ascii','ignore')
            docs.append(line.replace('\n', ''))
            doc_count+=1
            if doc_count==500000:
                return docs
    return docs

def write_clusters(docs, labels):
    """
    Write kmeans clusters to clusters.json file. 
    """
    clusters = defaultdict(list)
    for doc, label in zip(docs, labels):
        clusters[str(label)].append(doc)
    with open('clusters.json', 'w') as clusters_file:
        json.dump(clusters, clusters_file, ensure_ascii=False, indent = 4)

if __name__=='__main__':
	docs = read_phrases('phrases.gz')
	n_clusters = 1000
	hasher = HashingVectorizer(n_features=90000,
	                                   stop_words='english', non_negative=True,
	                                   norm=None, binary=True)

	vectorizer = make_pipeline(hasher, TfidfTransformer())
	X = vectorizer.fit_transform(docs)
	model =  KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
	model.fit(X)
	write_clusters(docs, model.labels_)
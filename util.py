import json 
import gzip


def process_raw_phrases(file_path):
	"""
	Read phrases from compressed file and write it to phrases.gz file after pre-processing.
	"""
    with gzip.open('phrases.gz', 'w') as out_file:
        with gzip.open(file_path, 'r') as gz_file:
            out_file.write('cluster_id\troot\tphrase\tcount\n'.encode('ascii', 'ignore'))
            cluster_id = ''
            is_root = False
            cluster_count = 0
            line_count = 0
            for line in gz_file:
                line_count+=1
                if(line_count<5):
                    continue
                line = line.decode('ascii', 'ignore')
                ls = line.replace('\r\n','').split('\t')
                if len(ls)<2 or (ls[0]=='' and ls[1]==''):
                    #skip urls 
                    continue
                if ls[0] == '':
                    phrase = ls[3]
                    count = ls[1]
                    s = cluster_id+'\t'+'False'+'\t'+phrase+'\t'+count+'\n'
                    out_file.write(s.encode('ascii', 'ignore'))
                else:
                    cluster_id = ls[3]
                    phrase = ls[2]
                    count = ls[1]
                    s = cluster_id+'\t'+'True'+'\t'+phrase+'\t'+count+'\n'
                    out_file.write(s.encode('ascii', 'ignore'))
                    cluster_count+=1
                 
            print (cluster_count)  

if __name__=='__main__':
	# Download data from http://snap.stanford.edu/memetracker/srcdata/clust-qt08080902w3mfq5.txt.gz
	# process this data and write to phrases.gz compressed file. 
	process_raw_phrases('clust-qt08080902w3mfq5.txt.gz')

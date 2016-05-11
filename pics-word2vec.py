import re, getopt, sys, os
from word2vec_retriever import Word2VecRetriever
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def usage():
	print ("Usage: {} [-f/--vector-file (vector file)]\n"
		"\t[-q/--query-file]\n"
		"\t[-s/--keep-stopwords (keep stopwords)]\n"
		"\t[-t/--tf-idf (use TF-IDF weighting)]".format(sys.argv[0]))

def processArgs():
	removeStopWords = False
	useTfIdf = False
	vectorFile = None
	queryFile = "data_4stdpt/queries_val"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "f:q:sth",
			["vector-file=", "query-file=", "keep-stopwords", "tf-idf", "help"])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-s", "--remove-stopwords"):
			removeStopWords = True
		elif o in ("-f", "--vector-file"):
			if not os.path.isfile(a):
				print("Invalid gloVec vector file [{}]".format(a))
				usage()
				sys.exit(1)
			vectorFile = a
		elif o in ("-q", "--query-file"):
			if not os.path.isfile(a):
				print("Invalid query file [{}]".format(a))
				usage()
				sys.exit(1)
			queryFile = a
		elif o in ("-t", "--tf-idf"):
			useTfIdf = True
		else:
			assert False, "unhandled option"

	if vectorFile is None:
		usage()
		sys.exit(1)

	return { "queryFile" : queryFile, "useTfIdf" : useTfIdf,
		"removeStopWords" : removeStopWords, "vectorFile" : vectorFile }

args = processArgs()
retriever = Word2VecRetriever(args['removeStopWords'], args['vectorFile'], 
	args['useTfIdf'])
retriever.loadStopWords("stopwords_en.txt")
retriever.loadVectors()
retriever.build_documents("data_4stdpt/target_collection")

results = retriever.process_query_file(args['queryFile'])
resHandler = open("pics-word2vec-res.txt", "w")
retriever.writeResults(resHandler)
resHandler.close()

sys.exit(0)

map1000 = retriever.mapk()
recall, precision = retriever.getInterpolatedPrecisionRecall()

swStr = "sw"
if retriever.getStopWords():
	swStr = "nsw"
tfStr = "std"
if retriever.getTfIdf():
	tfStr = "tfidf"
output = open("pics-word2vec-{}-{}-{}.txt".format(swStr, tfStr, os.getpid()), "w")
output.write("# MAP = {}\n".format(map1000))
output.write("# Recall\tPrecision\n")
for r, p in zip(recall, precision):
	output.write("{}\t{}\n".format(r,p))
output.close()

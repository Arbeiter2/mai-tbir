import re, getopt, sys, os
from glove_retriever import GloVecRetriever


def usage():
	print ("Usage: {} -v/--vector-length=[50|100|200|300]\n"
		"\t[-s/--keep-stopwords (keep stopwords)]\n"
		"\t[-f/--vector-file (vector file)]\n"
		"\t[-t/--tf-idf (use TF-IDF weighting)]".format(sys.argv[0]))

def processArgs():
	removeStopWords = False
	useTfIdf = False
	vecLength = 0
	vectorFile = None

	try:
		opts, args = getopt.getopt(sys.argv[1:], "v:f:sth",
			["vector-length=", "vector-file=", "keep-stopwords", "tf-idf", "help"])
	except getopt.GetoptError as err:
		print(err)
		usage()
		sys.exit(2)

	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-v", "--vector-length"):
			a = int(a)
			if a not in [50, 100, 200, 300]:
				print("Invalid gloVec vector length [{}]".format(a))
				usage()
				sys.exit(1)
			vecLength = a
		elif o in ("-s", "--remove-stopwords"):
			removeStopWords = True
		elif o in ("-f", "--vector-file"):
			if not os.path.isfile(a):
				print("Invalid gloVec vector file [{}]".format(a))
				usage()
				sys.exit(1)
			vectorFile = a
		elif o in ("-t", "--tf-idf"):
			useTfIdf = True
		else:
			assert False, "unhandled option"

	if vecLength == 0 or vectorFile is None:
		usage()
		sys.exit(1)

	return { "vecLength" : vecLength, "useTfIdf" : useTfIdf,
		"removeStopWords" : removeStopWords, "vectorFile" : vectorFile }

args = processArgs()
retriever = GloVecRetriever(args['removeStopWords'], args['vecLength'],
	args['vectorFile'], args['useTfIdf'])
retriever.loadStopWords("stopwords_en.txt")
retriever.loadVectors()
retriever.build_documents("data_4stdpt/target_collection")
actual, results = retriever.process_query_file("data_4stdpt/queries_val")

map1000 = retriever.mapk()
recall, precision = retriever.getInterpolatedPrecisionRecall()

swStr = "sw"
if retriever.getStopWords():
	swStr = "nsw"
tfStr = "std"
if retriever.getTfIdf():
	tfStr = "tfidf"
output = open("pics-glove-{}-{}-{}-{}.txt".format(retriever.getVectorLength(),
	swStr, tfStr, os.getpid()), "w")
output.write("# MAP = {}\n".format(map1000))
output.write("# Recall\tPrecision\n")
for r, p in zip(recall, precision):
	output.write("{}\t{}\n".format(r,p))
output.close()

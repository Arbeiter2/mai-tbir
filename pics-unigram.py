from unigramretriever import UnigramRetriever
from average_precision import mapk
import sys, getopt, os

# process command line arguments
def processArgs():
	removeStopWords = False
	doStemming = False
	queryFile = "data_4stdpt/queries_val"

	try:
		opts, args = getopt.getopt(sys.argv[1:], "sq:th",
			["remove-stopwords", "query-file=", "do-stemming", "help"])
	except getopt.GetoptError as err:
		print(err)
		usage() 
		sys.exit(2)

	for o, a in opts:
		if o in ("-h", "--help"):
			usage()
			sys.exit()
		elif o in ("-q", "--query-file"):
			if not os.path.isfile(a):
				print("Invalid query file [{}]".format(a))
				usage()
				sys.exit(1)
			queryFile = a
		elif o in ("-s", "--remove-stopwords"):
			removeStopWords = True
		elif o in ("-t", "--do-stemming"):
			doStemming = True
		else:
			usage()
			assert False, "unhandled option"

	return { 'removeStopWords' : removeStopWords, 'doStemming' : doStemming,
		"queryFile" : queryFile }

# show usage message
def usage():
	print ("Usage: {} [-h/--help (help)]\n\t"
		"[-q/--query-file]\n\t"
		"[-s/--remove-stopwords (remove stopwords)]\n\t"
		"[-t/--do-stemming (apply stemming to words)]".format(sys.argv[0]))


args = processArgs()
retriever = UnigramRetriever(args['doStemming'], args['removeStopWords'])
retriever.loadStopWords("stopwords_en.txt")
retriever.build_documents("data_4stdpt/target_collection")
#actual, results = retriever.process_query_file("data_4stdpt/queries_val")
results = retriever.process_query_file(args['queryFile'])

resHandler = open("pics-unigram-res.txt", "w")
retriever.writeResults(resHandler)
resHandler.close()

sys.exit(0)

print(retriever.mapk())
recall, precision = retriever.getInterpolatedPrecisionRecall()

if args['removeStopWords']:
	swStr = "nsw"
else:
	swStr = "sw"
if args['doStemming']:
	stStr = "st"
else:
	stStr = "nst"
output = open("pics-unigram-{}-{}-{}.txt".format(swStr, stStr, os.getpid()), "w")
output.write("# Recall\tPrecision\n")
for r, p in zip(recall, precision):
	output.write("{}\t{}\n".format(r,p))
output.close()

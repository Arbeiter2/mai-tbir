import os, sys, getopt, re
from nltk.stem.lancaster import LancasterStemmer
from average_precision import mapk
import numpy as np
from rankedresultseval import recall, precision, interpolated_precision, precision_at_fixed_recall


class TextRetriever:

	def __init__(self, doStemming, removeStopWords):
		self.stemmer = LancasterStemmer()
		self.doStemming = doStemming
		self.removeStopWords = removeStopWords
		self.docsLoaded = False
		self.stopWords = []
		self.groundTruth = []
		self.results = []
		self.documents = {}

	# convert ranked list of results to a list of tuples, of the format
	# [(1, [0|1]), (2, [0|1]) .... ]
	# where the first tuple element is the result rank and the second is the relevance
	# and 0=not relevant, 1=relevant
	def build_serp(self, res, relevant):
		serp = []
		for i, r in enumerate(res):
			q = 0
			if r in relevant:
				q = 1
			serp.append((i, q))
		return serp

	# load English stop words
	def loadStopWords(self, filename):
		self.stopWords = []
		f = open(filename, "r")
		for w in f:
			self.stopWords.append(w.strip())
		f.close()

	# tokenise a query or sentence
	def tokenise(self, sentence):
		words = re.sub('[^a-zA-Z0-9\'\s]+', " ", sentence.lower())
		words = words.strip()
		#words = list(map(lambda x: re.sub('[^a-zA-Z0-9\-\'\s]+', "", x),
			#sentence))
		words = re.split('\s+', words)
		if self.removeStopWords:
			words = list(filter(lambda x: x not in self.stopWords, words))
		if self.doStemming:
			words = list(map(self.stemmer.stem, words))

		return words

	# build models, index or other structures for this corpus
	# documents are of the form
	# <sentenceID>\t<docID>\t<sentence>
	# sets self.docsLoaded to True on completion
	def build_documents(self, filename):
		fo = open(filename, "r")
		self.wordCount = 0
		for line in fo:
			line = line.strip()
			(sent_id, docID, sentence) = line.split('\t')

			if not re.match('^\d+$', sent_id):
				continue

			if docID not in self.documents:
				self.documents[docID] = { 'sentence' : sentence, 'models' : [] }
			model, cnt = self.build_model(sentence, docID)
			self.documents[docID]['models'].append(model)
			self.wordCount += cnt



	# run query against document models and return ranked list of results
	# results are of the format
	# [{ 'docID' : docID1 }, { 'docID' : docID2 } ... ]
	# other object members may be added as required
	def run_query(self, sentence, topN=1000, smoothing=0.5):
		raise NotImplementedError("run_query(...) not implemented")

	# return an object representation of the sentence model, linked to parent docID
	def build_model(self, sentence, parent):
		raise NotImplementedError("build_model(...) not implemented")

	# read in file of queries, and run each query against already loaded documents
	# input file has format 
	# <sentenceID>\t<docID>\t<sentence>
	# builds self.groundTruth as a list of lists, with the following structure
	# self.groundTruth = [docID] * number of occurences
	def process_query_file(self, queryFile, topN=1000):
		assert self.docsLoaded

		self.results = []
		self.groundTruth = []
		test = open(queryFile, "r")
		for line in test:
			line = line.strip()
			(sent_id, docID, query) = line.split('\t')

			if not re.match('^\d+$', sent_id):
				continue

			self.groundTruth.append([docID] * len(self.documents[docID]['models']))
			prediction = self.run_query(query, topN)
			self.results.append(prediction)
		test.close()

		return self.groundTruth, self.results

	def mapk(self, topN=1000):
		return 100 * mapk(self.groundTruth, self.results, topN)

	def getStopWordList(self):
		return self.stopWords

	def getStopWords(self):
		return self.removeStopWords
	
	def getStemming(self):
		return self.doStemming
	
	def getGroundTruth(self):
		return self.groundTruth

	def getPredictions(self):
		return self.results

	
	def getInterpolatedPrecisionRecall(self):
		out = np.zeros(11)	# for 0.0, 0.1, ... 1.0
		recall_pts = [float(x)/10 for x in range(0, 11)]
		for actual, result in zip(self.groundTruth, self.results):
			serp = self.build_serp(result, actual)
			rec = recall(serp, len(actual))
			prec = precision(serp)
			interp_prec = interpolated_precision(serp, prec)
			unused, prf = np.array(precision_at_fixed_recall(interp_prec, rec, recall_pts))
			out += np.array(prf)

		return recall_pts, out/len(self.results)



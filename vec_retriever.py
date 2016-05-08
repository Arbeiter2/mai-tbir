from textretriever import TextRetriever
from collections import Counter
import math, sys
import re, getopt
from operator import itemgetter
import numpy as np
import pandas as pd
from numpy import linalg as LA


class VectorRetriever(TextRetriever):
	def __init__(self, removeStopWords, vecLength, vectorFile, useTfIdf):
		TextRetriever.__init__(self, False, removeStopWords)
		self.vecLength = vecLength
		self.vectorFile = vectorFile
		self.vocab = {}
		self.documents = {}
		self.useTfIdf = useTfIdf
		self.vectors = None
		self.index = dict()
		self.idf = dict()


	# loads self.vectors
	# implementation specific
	def loadVectors(self):
		 raise NotImplementedError("loadVectors(...) not implemented")

	# builds self.documents, self.index and self.vocab
	def build_documents(self, filename):
		fo = open(filename, "r")
		docCount = 0.0 
		for line in fo: 
			line = line.strip()
			(sent_id, docID, sentence) = line.split('\t')

			if not re.match('^\d+$', sent_id):
				continue
			docCount = docCount + 1.0 
			tokens = self.tokenise(sentence)

			# keep only words in vocab
			tokens = list(filter(lambda x: x in self.vocab, tokens))

			if docID not in self.documents:
				self.documents[docID] = { 'models' : [] }

			m = { 'sentence' : sentence }
			if self.useTfIdf:
				# calculate tf 
				tf = dict(Counter(tokens))
				sum = np.sum(tf.values())
				for i in tf: 
					tf[i] = (0.0 + tf[i])/sum

				# add parent document to index
				for w in tf: 
					if w not in self.index:
						self.index[w] = set()
					self.index[w].add(docID)
				m['tf'] = tf
			else:
				m['vector'] = np.zeros(self.vecLength)
				for w in tokens:
					if w not in self.index:
						self.index[w] = set()
					self.index[w].add(docID)
					m['vector'] += self.getVec(w)

			self.documents[docID]['models'].append(m)

		fo.close()
		
		self.docsLoaded = True

		if not self.useTfIdf:
			return

		# calculate idf
		for w in self.index:
			self.idf[w] = (1.0 * docCount)/len(self.index[w])

		for docID in self.documents:
			for m in self.documents[docID]['models']:
				if self.useTfIdf:
					m['vector'] = np.zeros(self.vecLength)
					for w in m['tf']:
						m['vector'] += self.getVec(w) * m['tf'][w] * self.idf[w]
					#m['vector'] = v/len(m['tf'])
				#m['vector'] =  m['vector']/LA.norm(v)


	# return vector representation of word
	# implementation specific
	def getVec(self, word):
		raise NotImplementedError("getVec(...) not implemented")

	# find topN documents, ordered by decreasing score,
	# most closely matching the sentence/query
	def run_query(self, sentence, topN=1000, smoothing=0.5):
		words = self.tokenise(sentence)
		queryVec = np.zeros(self.vecLength)
		candidates = set()

		if self.useTfIdf:
			# calculate tf of query terms
			tf = dict(Counter(words))
			sum = np.sum(tf.values())
			for i in tf:
				tf[i] = (1.0 * tf[i])/sum

			# build a set of candidate documents
			for w in tf:
				if w not in self.index:
					continue
				candidates = candidates.union(self.index[w])
				#print(w, tf[w], idf[w], tf[w] * idf[w])
				queryVec = queryVec + self.getVec(w) * tf[w] * self.idf[w]
		else:
			# build a set of candidate documents
			for w in words:
				if w in self.index:
					candidates = candidates.union(self.index[w])
					queryVec = queryVec + self.getVec(w)

		#queryVec /= (LA.norm(queryVec)/len(words))

		best_matches = []
		for docID in candidates:
			for m in self.documents[docID]['models']:
				cosine = self.cosine_sim(queryVec, m['vector'])
				best_matches.append({ 'docID' : docID, 'dist' : cosine,
					'model' : m })

		# sort best_matches by logProb
		MLE = sorted(best_matches, key=itemgetter('dist'), reverse=True)

		# return docIDs of top N entries
		return [x['docID'] for x in MLE[:topN]]


	# find cosine distance between two vectors
	def cosine_sim(self, a, b):
		x = a.dot(b)
		return x/(np.linalg.norm(a) * np.linalg.norm(b))

	def getVectorLength(self):
		return self.vecLength

	def getTfIdf(self):
		return self.useTfIdf


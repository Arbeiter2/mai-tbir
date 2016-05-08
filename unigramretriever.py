from textretriever import TextRetriever
import math
import re
from operator import itemgetter


class UnigramRetriever(TextRetriever):
	def __init__(self, doStemming, removeStopWords):
		TextRetriever.__init__(self, doStemming, removeStopWords)
		self.vocab = {}

	def build_documents(self, filename):
		self.vocab = {}
		TextRetriever.build_documents(self, filename)


		# calculate marginal prob for all words
		for i in self.vocab:
			self.vocab[i]['prob'] /= self.wordCount

		self.docsLoaded = True



	# find topN documents, ordered by decreasing score,
	# most closely matching the sentence/query
	def run_query(self, sentence, topN=1000, smoothing=0.5):
		words = self.tokenise(sentence)

		# build a set of candidate documents
		candidates = set()
		for w in words:
			if w in self.vocab:
				candidates = candidates.union(self.vocab[w]['docs'])

		best_matches = []
		for docID in candidates:
			for m in self.documents[docID]['models']:
				total = 0.0
				for qw in words:
					tokenProb = ((smoothing * self.getDocProb(m, qw)) +
						((1.0 - smoothing) * self.getGlobalProb(qw)))
					if tokenProb > 0.0:
						total = total + math.log10(tokenProb)

				best_matches.append({ 'docID' : docID, 'logProb' : total,
					'model' : m })

		# sort best_matches by logProb
		MLE = sorted(best_matches, key=itemgetter('logProb'), reverse=True)

		# return docIDs of top N entries
		return [x['docID'] for x in MLE[:topN]]

	# get the probability of token in given model
	def getDocProb(self, model, token):
		if token not in model:
			return 0.0
		else:
			return model[token]


	# get the probability of token in total vocabulary
	def getGlobalProb(self, token):
		if token not in self.vocab:
			return 0.0
		else:
			return self.vocab[token]['prob']

	# build a unigram model of a sentence
	def build_model(self, sentence, parent):
		model = dict()
		total = 0
		tokens = self.tokenise(sentence)
		for w in tokens:
			if w not in model:
				model[w] = 0
			if w not in self.vocab:
				self.vocab[w] = { 'prob' : 0.0, 'docs' : set() }
			model[w] = model[w] + 1.0
			self.vocab[w]['prob'] += 1.0

			total = total + 1

		for token in model:
			model[token] /= total
			self.vocab[token]['docs'].add(parent)
		return (model, len(tokens))


from vec_retriever import VectorRetriever
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

class Word2VecRetriever(VectorRetriever):
	def __init__(self, removeStopWords, vectorFile, useTfIdf):
		VectorRetriever.__init__(self, removeStopWords, 300, vectorFile, useTfIdf)

	#
	# override of abstract method VectorRetriever.loadVectors()
	#
	def loadVectors(self):
		assert (self.vectorFile is not None), "Bad vector length or filename"

		self.vectors = Word2Vec.load_word2vec_format(self.vectorFile, binary=True)
		self.vectors.init_sims(replace=True)

		self.vocab = self.vectors

	# override of getVec
	def getVec(self, word):
		return self.vectors[word]


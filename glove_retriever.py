from textretriever import TextRetriever
from vec_retriever import VectorRetriever
import math, sys
import re, getopt
from operator import itemgetter
import numpy as np
import pandas as pd
from numpy import linalg as LA


class GloVecRetriever(VectorRetriever):
	def __init__(self, removeStopWords, vecLength, vectorFile, useTfIdf):
		VectorRetriever.__init__(self, removeStopWords, vecLength, vectorFile, useTfIdf)

	#
	# override of abstract method VectorRetriever.loadVectors()
	#
	def loadVectors(self):
		assert (self.vecLength != 0
			and self.vectorFile is not None), "Bad vector length or filename"

		reader = pd.read_csv(self.vectorFile, header=None, delimiter=r"\s+",
			low_memory=False, chunksize=16*1024, index_col=0, encoding='utf-8')

		self.vectors = pd.DataFrame()
		for chunk in reader:
			self.vectors = self.vectors.append(chunk)
		self.vocab = self.vectors.index.values

		print("Vector file loaded")

	# override of getVec
	def getVec(self, word):
		return self.vectors.loc[word]


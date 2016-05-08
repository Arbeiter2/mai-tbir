from gensim.models import Word2Vec
import os
import sys
import resource

resource.getrlimit(resource.RLIMIT_MEMLOCK)

def usage():
	print ("{} <vector-file-path>".format(sys.argv[0]))

reload(sys)
sys.setdefaultencoding('utf8')

if len(sys.argv) < 2 or not os.path.isfile(sys.argv[1]):
	usage()
	sys.exit(1)

model = Word2Vec.load_word2vec_format(sys.argv[1], binary=True)
model.init_sims(replace=True)

fname = 'questions-words.txt'
ename = 'w2v-{}-stats.txt'.format(os.getpid())
right = wrong = 0

stats = open(ename, "w")

res = model.accuracy(fname)
for section in res:
	right = right + len(section['correct'])
	wrong = wrong + len(section['incorrect'])
	if  len(section['correct']) +  len(section['incorrect']) > 0:
		print("MAP@1 [{}] : {}".format(section['section'], float(len(section['correct']))/(len(section['correct']) + len(section['incorrect']))))
print("{},{}/{}\n".format(right, wrong, (1.0 * right)/(right + wrong)))


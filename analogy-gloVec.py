from scipy.spatial.distance import cosine
import sys, getopt
import os, getopt
import numpy as np
import pandas as pd
import csv

reload(sys)
sys.setdefaultencoding('utf8')
csv.field_size_limit(500 * 1024 * 1024)

def accuracy(correct, incorrect):
	if (correct + incorrect) == 0:
		return 0.0
	return float(correct)/(correct + incorrect)

def cosine_sim(a, b):
	x = a.dot(b)
	return x/(np.linalg.norm(a) * np.linalg.norm(b))

def most_similar(positive=None, negative=None, topn=10):
	if positive is None and negative is None:
		return None

	# Build a "mean" vector for the given positive and negative terms
	mean = pd.DataFrame(vec.iloc[0])
	mean = 0.0
	for word in positive: mean += vec.loc[word]
	for word in negative: mean += (-1 * vec.loc[word])
	
	mean_norm = np.sqrt(np.square(mean).sum(axis=1))
	mean /= mean_norm
	
	# Now calculate cosine distances between this mean vector and all others
	dists = vec.dot(mean)
	dists /= (vec_norm * mean_norm)

	best = dists[~dists.index.isin(positive + negative)].idxmax()
	
	return (best, dists.loc[dists.idxmax()])


def usage():
	print("Usage: {} -v/--vector-length=[50|100|200|300]\n"
		"\t-f/--vector-file=<vector-file>".format(sys.argv[0]))

# check command line args
try:
	opts, args = getopt.getopt(sys.argv[1:], "v:f:h",
		["vector-length=", "vector-file=", "help"])
except getopt.GetoptError as err:
	print(err)
	usage()
	sys.exit(2)

vecLength = None
vecFile = None

for o, a in opts:
	if o in ("-h", "--help"):
		usage()
		sys.exit()
	elif o in ("-v", "--vector-length"):
		vecLength = int(a)
		if vecLength not in [50, 100, 200, 300]:
			usage()
			sys.exit(1)
	elif o in ("-f", "--vector-file"):
		if not os.path.isfile(a):
			usage()
			sys.exit(1)
		else:
			vecFile = a
	else:
		usage()
		assert False, "unhandled option"

if vecLength is None or vecFile is None:
	usage()
	sys.exit(1)

reader = pd.read_csv(vecFile,
	header=None, delimiter=r"\s+", low_memory=False, chunksize=16*1024,
	index_col=0, encoding='utf-8')

vec = pd.DataFrame()
for chunk in reader:
	vec = vec.append(chunk)
vec_norm = np.sqrt(np.square(vec).sum(axis=1))

fname = 'questions-words.txt'
ename = 'gloVe-{}-{}-stats.txt'.format(sys.argv[2], os.getpid())
correct = incorrect = 0

vocab = vec.index.values
sections = { }

ftab = open(fname, 'r')
stats = open(ename, 'w')
for line in ftab:
	# Beijing China Madrid Spain
	fields = [x.lower() for x in line.strip().split(' ')]
	#print(fields)
	if fields[0][0] == ':':
		section_name = fields[1]
		sections[section_name] = { 'correct' : 0, 'incorrect' : 0 }
		continue

	if (fields[0] not in vocab or
	    fields[1] not in vocab or
	    fields[2] not in vocab):
		continue

	#sys.stdout.write('.')
	r = most_similar(positive=[fields[2], fields[1]], negative=[fields[0]])
	if (r[0] == fields[3]):
		correct = correct + 1
		sections[section_name]['correct'] = (
			sections[section_name]['correct'] + 1)
	else:
		incorrect = incorrect + 1
		sections[section_name]['incorrect'] = (
			sections[section_name]['incorrect'] + 1)

		s = "{}|{} = {}; {} = {} ({})".format(section_name,
			fields[0], fields[1], fields[2], fields[3], r[0])
		stats.write(s + "\n")
		#print("\n" + s)
	if (correct + incorrect) % 1000 == 0:
		s="S|{}: {}/{}".format(correct + incorrect, correct, incorrect)
		#stats.write(s + "\n")
		print(s)
ftab.close();

stats.write("S|{}: {}/{}\n".format(correct + incorrect, correct, incorrect))

for s in sections:
	line = "Recall@1 [{}] = {}".format(s, accuracy(sections[s]['correct'],
		sections[s]['incorrect']))
	stats.write(line + "\n")
	print(line)

line = "Recall@1 [global] = {}".format(accuracy(correct, incorrect))
stats.write(line + "\n")
print(line)

stats.close();

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.datasets import base as bunch
from sklearn.datasets import load_files
from time import time
import pickle

def remove_header_subject(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    sub = [l for l in _before.split("\n") if "Subject:" in l]
    final = sub[0] + "\n" + after
    return final

sep = '-' * 50
AVI_HOME = './SRAA/partition1/data'
percent = 1./3
rnd = 2342
vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))

t0 = time()
print "Loading the SRAA data..."
data = load_files(AVI_HOME, encoding="latin1", load_content=True, random_state=rnd)
data.data = [remove_header_subject(text) for text in data.data]
print "Data loaded in %f." % (time() - t0)

print 'Total number of data: %d' % len(data.data)

indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, indices=True, random_state=rnd)
for train_ind, test_ind in indices:
    data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind]),
                          test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind]))

t0 = time()
print sep
print "Extracting features from the training dataset using a sparse vectorizer..."
print "Feature extraction technique is %s." % vect
X_tr = vect.fit_transform(data.train.data)
y_tr = data.train.target
duration = time() - t0
print "done in %fs" % duration
print "n_samples: %d, n_features: %d" % X_tr.shape

t0 = time()
print sep
print "Extracting features from the test dataset using the same vectorizer..."
X_te = vect.transform(data.test.data)
y_te = data.test.target
duration = time() - t0
print "done in %fs" % duration
print "n_samples: %d, n_features: %d" % X_te.shape

print "Saving X_train, y_train, X_test, y_test, X_train_corpus, X_test_corpus using pickle..."
pickle.dump(X_tr, open('SRAA_X_train.pickle', 'wb'))
pickle.dump(y_tr, open('SRAA_y_train.pickle', 'wb'))
pickle.dump(X_te, open('SRAA_X_test.pickle', 'wb'))
pickle.dump(y_te, open('SRAA_y_test.pickle', 'wb'))
pickle.dump(data.train.data, open('SRAA_X_train_corpus.pickle', 'wb'))
pickle.dump(data.test.data, open('SRAA_X_test_corpus.pickle', 'wb'))

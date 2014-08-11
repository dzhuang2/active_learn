from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.datasets import base as bunch
from sklearn.datasets import load_files

def remove_header_subject(text):
    """
    Given text in "news" format, strip the headers, by removing everything
    before the first blank line.
    """
    _before, _blankline, after = text.partition('\n\n')
    sub = [l for l in _before.split("\n") if "Subject:" in l]
    final = sub[0] + "\n" + after
    return final

def load_SRAA(avihome='./SRAA/partion1', percent=0.8, rnd=2342, \
              vect=CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))):
    data = load_files(AVI_HOME, encoding="latin1", load_content=True, random_state=rnd)
    data.data = [remove_header_subject(text) for text in data.data]

    indices = ShuffleSplit(len(data.data), n_iter=1, test_size=percent, indices=True, random_state=rnd)
    for train_ind, test_ind in indices:
        data = bunch.Bunch(train=bunch.Bunch(data=[data.data[i] for i in train_ind], target=data.target[train_ind]),
                              test=bunch.Bunch(data=[data.data[i] for i in test_ind], target=data.target[test_ind]))

    X_tr = vect.fit_transform(data.train.data)
    y_tr = data.train.target

    X_te = vect.transform(data.test.data)
    y_te = data.test.target
    
    return (X_tr, y_tr, X_te, y_te, data.train.data, data.test.data)
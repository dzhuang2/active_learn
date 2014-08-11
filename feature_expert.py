'''
feature_expert.py

This file hosts feature_expert related functions.
'''
from time import time
from imdb import load_imdb, load_newsgroups
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

class feature_expert(object):
    '''
    feature expert returns what it deems to be the most informative feature
    given a document
    
    feature expert ranks the features using one of the following criteria:
        1. mutual information
        2. logistic regression with L1 regularization weights
    
    '''
    def __init__(self, X, y, metric, smoothing=1e-6, C=0.1):
        self.sample_size, self.num_features = X.shape
        self.metric = metric
        self.smoothing = smoothing
        self.feature_rank = ([], [])
        
        print '-' * 50
        print 'Starting Feature Expert Training ...'
        start = time()
        
        if metric == 'mutual_info':
            self.feature_rank = self.rank_by_mutual_information(X, y)
        elif metric == 'L1':
            self.feature_rank = self.rank_by_L1_weights(C, X, y)
        else:
            raise ValueError('metric must be one of the following: \'mutual_info\', \'L1\'')
        
        print 'Feature Expert has deemed %d words to be of label 0' % len(self.feature_rank[0])
        print 'Feature Expert has deemed %d words to be of label 1' % len(self.feature_rank[1])
        
        print 'Feature Expert trained in %0.2fs' % (time() - start)
    
    def class0_features_by_rank(self):
        return self.feature_rank[0]
    
    def class1_features_by_rank(self):
        return self.feature_rank[1]
    
    def info(self):
        print 'feature expert is trained from %d samples on % features using metric \'%s\'' % \
            (self.sample_size, self.num_features, self.metric)
    
    def rank_by_mutual_information(self, X, y):
        self.feature_count = self.count_features(X, y)
        self.feature_mi_scores = np.zeros(shape=self.num_features)
        
        for f in range(self.num_features):
            probs = self.feature_count[f] / self.feature_count[f].sum()
            f_probs = probs.sum(1)
            y_probs = probs.sum(0)
            
            for i in range(2):
                for j in range(2):
                    self.feature_mi_scores[f] += probs[i,j]*(np.log2(probs[i,j]) - np.log2(f_probs[i])
                                                 - np.log2(y_probs[j]))
        
        feature_rank = np.argsort(self.feature_mi_scores)[::-1]

        return self.classify_features(feature_rank)
    
    def rank_by_L1_weights(self, C, X, y):
        clf_l1 = linear_model.LogisticRegression(C=C, penalty='l1')
        clf_l1.fit(X, y)
        self.L1_weights = clf_l1.coef_[0]
        self.feature_count = self.count_features(X, y)
        
        feature_rank = np.argsort(np.absolute(self.L1_weights))[::-1]

        return self.classify_features(feature_rank)
    
    def classify_features(self, feature_rank):
        class0_features_rank = list()
        class1_features_rank = list()
        
        for f in feature_rank:
            if self.feature_count[f,1,0] > self.feature_count[f,1,1]:
                class0_features_rank.append(f)
            elif self.feature_count[f,1,0] < self.feature_count[f,1,1]:
                class1_features_rank.append(f)
            # if positive and negative counts are tied, the feature is deemed
            # neither positive nor negative
        
        return (class0_features_rank, class1_features_rank)
    
    def count_features(self, X, y):
        X_csc = X.tocsc()
        feature_count = np.zeros(shape=(self.num_features, 2, 2))
        
        for f in range(self.num_features):
            feature = X_csc.getcol(f)
            nonzero_fi = feature.indices
            y_1 = np.sum(y)
            y_0 = len(y) - y_1
            feature_count[f,1,1] = np.sum(y[nonzero_fi])
            feature_count[f,1,0] = len(nonzero_fi) - feature_count[f,1,1]
            feature_count[f,0,0] = y_0 - feature_count[f,1,0]
            feature_count[f,0,1] = y_1 - feature_count[f,1,1]
            feature_count[f] += self.smoothing
        
        return feature_count
        
    def most_informative_feature(self, X, label):
        try:
            f = self.top_n_features(X, label, 1)[0]
        except IndexError:
            f = None
        return f
    
    def top_n_features(self, X, label, n):
        features = X.indices
        
        top_features = list()
        for f in self.feature_rank[int(label)]:
            if f in features:
                top_features.append(f)
            if len(top_features) == n:
                break
        
        return top_features
    
    def top_n_class0_features(self, X, n):
        return self.top_n_features(X, 0, n)
    
    def top_n_class1_features(self, X, n):
        return self.top_n_features(X, 1, n)
        
def write_features(filename='feature.txt'):
    with open(filename, 'w') as f:
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
        (X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs) = \
            load_imdb("./aclImdb", shuffle=True, vectorizer=vect)
        feature_names = vect.get_feature_names()
        for feature in feature_names:
            f.write(feature.encode('utf8') + '\n')
    return

def read_features(filename='feature.txt'):
    feature_names = list()
    with open(filename, 'r') as f:
        for line in f:
            feature_names.append(line.strip())
    return feature_names

def print_features(feature_names, feature_expert, top_n, doc, X, y, X_text):
    class_label = {0:'negative', 1:'positive'}
    print 'Document #%d:' % doc
    print '-' * 50
    print X_text[doc]
    print '-' * 50
    print 'Instance Expert: %s' % class_label[y[doc]]
    
    top_n_features = feature_expert.top_n_features(X[doc], y[doc], top_n)
    top_n_features_str = [feature_names[f] for f in top_n_features]
    print 'Feature Expert Top(%d): %s' % (top_n, ', '.join(top_n_features_str))

def print_all_features(feature_names, feature_expert, top_n, doc, X, y, X_text):
    class_label = {0:'negative', 1:'positive'}
    print 'Document #%d:' % doc
    print '-' * 50
    print X_text[doc]
    print '-' * 50
    print 'Instance Expert: %s' % class_label[y[doc]]
    
    top_n_class0_features = feature_expert.top_n_class0_features(X[doc], top_n)
    top_n_class0_features_str = [feature_names[f] for f in top_n_class0_features]
    top_n_class1_features = feature_expert.top_n_class1_features(X[doc], top_n)
    top_n_class1_features_str = [feature_names[f] for f in top_n_class1_features]
    print 'Feature Expert Negative Top(%d): %s' % (top_n, ', '.join(top_n_class0_features_str))
    print 'Feature Expert Positive Top(%d): %s' % (top_n, ', '.join(top_n_class1_features_str))
    
def check_feature_expert(dataset='imdb', metric='mutual_info', top_n=10, smoothing=1e-6, C=0.1, \
                         vect=CountVectorizer(min_df=5, max_df=1.0, binary=False)):
    class_label = {0:'negative', 1:'positive'}
    if isinstance(dataset, str) and dataset == 'imdb':
        X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)
    elif isinstance(dataset, tuple) and len(dataset) == 3 and dataset[0] == 'newsgroup':
        X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = load_newsgroups(dataset[1], dataset[2], vectorizer=vect)
    
    feature_names = vect.get_feature_names()
    fe = feature_expert(X_pool, y_pool, metric, smoothing, C)
    doc_ids = np.random.permutation(np.arange(X_pool.shape[0]))
    
    print '\n'
    print '=' * 50
    
    for doc in doc_ids:
        print_all_features(feature_names, fe, top_n, doc, X_pool, y_pool, X_pool_docs)
        
        print '=' * 50
        ch = raw_input('Display the next document? Press Enter to continue or type \'n\' to exit...  ')
        
        if ch == 'n':
            break
    
    return
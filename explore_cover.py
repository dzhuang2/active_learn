'''
explore_cover.py

This file contains functions that investigates how much each word 
covers the imdb corpus.
'''
from time import time
import argparse
from active_learn import load_dataset
from feature_expert import feature_expert
from imdb import load_imdb, load_newsgroups
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

def covering(dataset, type='any', metric='L1', C=0.1, smoothing=1e-6):
    X_pool, y_pool, X_test, y_test = load_dataset(dataset)
    num_samples, num_feat = X_pool.shape
    
    fe = feature_expert(X_pool, y_pool, metric, smoothing, C)
    feature_count = np.zeros(num_feat)
    
    # no_feature_docs[0] counts # of documents labeled 0 but without any features
    # no_feature_docs[1] counts # of documents labeled 1 but without any features
    no_feature_docs = np.zeros(2)
    
    for doc in range(num_samples):
        label = y_pool[doc]
        
        if type == 'any':
            top_class0_feature = fe.top_n_class0_features(X_pool[doc], 1)
            top_class1_feature = fe.top_n_class1_features(X_pool[doc], 1)
            
            if len(top_class0_feature) == 0:
                no_feature_docs[0] += 1
            
            class0_feature_weight = fe.L1_weights[top_class0_feature]
            class1_feature_weight = fe.L1_weights[top_class1_feature]
            
            if np.absolute(class0_feature_weight) >= np.absolute(class1_feature_weight):
                top_feature = top_class0_feature
            else:
                top_feature = top_class1_feature
            
            feature_count[top_feature] += 1
            
        if type == 'all':
            feature_count[feature] += 1
        
    print 'number of features needed to cover the entire corpus = %d' % len(np.nonzero(feature_count)[0])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cat', default=['alt.atheism', 'talk.religion.misc'], nargs=2, \
                        help='2 class labels from the 20newsgroup dataset')
    parser.add_argument('-dataset', default='SRAA', help='dataset')
    parser.add_argument('-type', default='any', choices=['any', 'all'], help='class agnostic vs class sensitive')
    args = parser.parse_args()
    
    if args.dataset == '20newsgroups':
        covering((args.dataset, args.cat[0], args.cat[1]), args.type)
    else:
        covering(args.dataset, args.type)
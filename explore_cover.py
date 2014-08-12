'''
explore_cover.py

This file contains functions that investigates how much each word 
covers the imdb corpus.
'''
from time import time
import argparse
from active_learn import load_dataset
from feature_expert import feature_expert
import pickle
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
        
        if type == 'agnostic':
            top_class0_feature = fe.top_n_class0_features(X_pool[doc], 1)
            top_class1_feature = fe.top_n_class1_features(X_pool[doc], 1)
            
            if len(top_class0_feature) == 0 and len(top_class1_feature) == 0:
                no_feature_docs[label] += 1
            elif len(top_class0_feature) == 0 and len(top_class1_feature) != 0:
                # if there is no class 1 feature, then the top feature is the class0's top feature
                top_feature = top_class1_feature[0]
                feature_count[top_feature] += 1
            elif len(top_class0_feature) != 0 and len(top_class1_feature) == 0:
                # if there is no class 0 feature, then the top feature is the class1's top feature
                top_feature = top_class0_feature[0]
                feature_count[top_feature] += 1
            else:
                # if both classes have a valid top feature, then compare the absolute value of the weights
                # of both features to determine the top feature for this document
                class0_feature_weight = fe.L1_weights[top_class0_feature[0]]
                class1_feature_weight = fe.L1_weights[top_class1_feature[0]]
                
                if np.absolute(class0_feature_weight) >= np.absolute(class1_feature_weight):
                    top_feature = top_class0_feature[0]
                else:
                    top_feature = top_class1_feature[0]
                
                feature_count[top_feature] += 1
            
        elif type == 'sensitive':
            feature = fe.most_informative_feature(X_pool[doc], label)
            if feature == None:
                no_feature_docs[label] += 1
            else:
                feature_count[feature] += 1
            
    print 'number of features needed to cover the entire corpus = %d' % len(np.nonzero(feature_count)[0])
    print 'number of uncovered class 0 documents: %d' % no_feature_docs[0]
    print 'number of uncovered class 1 documents: %d' % no_feature_docs[1]
    pickle.dump(feature_count, open('feature_count.pickle', 'wb'))
    pickle.dump(no_feature_docs, open('uncovered_count.pickle', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cat', default=['alt.atheism', 'talk.religion.misc'], nargs=2, \
                        help='2 class labels from the 20newsgroup dataset')
    parser.add_argument('-dataset', default='SRAA', help='dataset')
    parser.add_argument('-allfeatures', action='store_true', help='use all features for L1')
    parser.add_argument('-type', default='sensitive', choices=['agnostic', 'sensitive'], help='class agnostic vs class sensitive')
    args = parser.parse_args()
    
    if args.allfeatures:
        metric = 'L1-count'
    else:
        metric = 'L1'
    if args.dataset == '20newsgroups':
        covering([args.dataset, args.cat[0], args.cat[1]], args.type, metric=metric)
    else:
        covering(args.dataset, args.type, metric=metric)
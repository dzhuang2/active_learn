'''
Created on Aug 15, 2014

Covering analysis for the features
'''

import numpy as np
import argparse
import pickle
from sklearn.linear_model import LogisticRegression
from imdb import load_newsgroups, load_nova
from sklearn.feature_extraction.text import CountVectorizer

def label_agnostic_covering(X, feature_rank):
    '''
    Finds out in how many documents each feature is the top feature,
    and how many documents do not have any useful feature.
    
    feature_rank contains the ids for the useful features,
    in decreasing order of usefulness.
    
    This code assumes that the features are discovered in the order
    of importance. In reality, a feature is discovered when a document
    in which it is the top feature is presented to the expert.
    '''
    
    feature_cover_counts = {}

    uncovered_docs = set(range(X.shape[0]))
    
    index = 0
    
    while len(uncovered_docs) > 0 and index < len(feature_rank):
        
        lp = list(uncovered_docs)
        
        candidate_X = X[lp].tocsc()
        
        # These are the docs where the feature_rank[index] occurs
        nzdocs = candidate_X[:, feature_rank[index]].indices
        
        nzdocs = np.array(lp)[nzdocs]
        
        if len(nzdocs) > 0:
            feature_cover_counts[feature_rank[index]] = len(nzdocs)
            uncovered_docs.difference_update(nzdocs)
        
        index += 1
    
    return feature_cover_counts, uncovered_docs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default=['imdb'], nargs='*', \
                    help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    args = parser.parse_args()
    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
    
    if args.dataset == ['imdb']:
        X_pool, y_pool, _, _, _, _ = load_imdb("../aclImdb", shuffle=True, vectorizer=vect)
    elif len(args.dataset) == 3 and args.dataset[0] == '20newsgroups':
        X_pool, y_pool, _, _, _, _ = load_newsgroups(args.dataset[1], args.dataset[2], shuffle=True, vectorizer=vect)
    elif args.dataset == ['SRAA']:
        X_pool = pickle.load(open('SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('SRAA_y_train.pickle', 'rb'))
    elif args.dataset == ['nova']:
        X_pool, y_pool, _, _, = load_nova()
    else:
        raise ValueError('Invalid Dataset!')
    
    if args.dataset != ['SRAA'] and args.dataset != ['nova']:
        feature_names = vect.get_feature_names()
    elif args.dataset == ['SRAA']:
        feature_names = pickle.load(open('SRAA_feature_names.pickle', 'rb'))
    # Note: nova dataset has no feature_names

    clf_l1 = LogisticRegression(penalty='l1', C=0.1)
    clf_l1.fit(X_pool, y_pool)
    l1_weights = clf_l1.coef_[0]
    
    # label agnostic
    non_zero_features = np.nonzero(l1_weights)[0]
    non_zero_features_ranked = non_zero_features[np.argsort(np.abs(l1_weights[non_zero_features]))[::-1]]
    
    fcc, ud = label_agnostic_covering(X_pool, non_zero_features_ranked)
    
    print '-' * 50
    print "COVERING ANALYSIS"
    print '-' * 50
    
    
    print
    print '-' * 50
    print "Label agnostic"
    print '-' * 50
    
    
    print "Non-zero feature count: %d" %(len(non_zero_features_ranked))
    
    print "Number of features required to cover all docs: %d" %(len(fcc))
    
    print "Number of features that do not cover any docs: %d" %(len(non_zero_features_ranked) - len(fcc))
    
    print "Uncovered doc count: %d" %(len(ud))
    
    
    #for fi in non_zero_features_ranked[:20]:
    #    print "%s:\t%d" %(feature_names[fi], fcc[fi])
    
    
    # label sensitive
    class0_features = np.nonzero(l1_weights < 0)[0]
    class1_features = np.nonzero(l1_weights > 0)[0]
    class0_features_ranked = class0_features[np.argsort(l1_weights[class0_features])]
    class1_features_ranked = class1_features[np.argsort(l1_weights[class1_features])[::-1]]
    
    class0_X = X_pool[np.arange(len(y_pool))[y_pool==0]]
    
    # We can use the same code by restricting X_pool to a specific class
    fcc, ud = label_agnostic_covering(class0_X, class0_features_ranked)
    
    print
    print '-' * 50
    print "Label sensitive"
    print '-' * 50
    
    print "Class 0"
    
    print "Non-zero class0 feature count: %d" %(len(class0_features_ranked))
    
    print "Number of class0 features required to cover all class0 docs: %d" %(len(fcc))
    
    print "Number of class0 features that do not cover any class0 docs: %d" %(len(class0_features_ranked) - len(fcc))
    
    print "Uncovered class0 doc count: %d" %(len(ud))
    
    
    
    #for fi in class0_features_ranked[:20]:
    #    print "%s:\t%d" %(feature_names[fi], fcc[fi])
    
    
    class1_X = X_pool[np.arange(len(y_pool))[y_pool==1]]
    
    # We can use the same code by restricting X_pool to a specific class
    fcc, ud = label_agnostic_covering(class1_X, class1_features_ranked)
    
    print
    
    print "Class 1"
    
    print "Non-zero class1 feature count: %d" %(len(class1_features_ranked))
    
    print "Number of class1 features required to cover all class1 docs: %d" %(len(fcc))
    
    print "Number of class1 features that do not cover any class1 docs: %d" %(len(class1_features_ranked) - len(fcc))
    
    print "Uncovered class1 doc count: %d" %(len(ud))
    
    #for fi in class1_features_ranked[:20]:
    #    print "%s:\t%d" %(feature_names[fi], fcc[fi])
    

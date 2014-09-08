'''
Created on Aug 14, 2014

Prints information about:

What is the number of all features?
What is the number of features that have non-zero weight (for a logistic regression that is trained on the full training dataset)?
What is the class distribution of the features that have non-zero weight?

'''

import argparse
import numpy as np

from feature_expert import feature_expert
from active_learn import load_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default=['imdb'], nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-metric', choices=['mutual_info', 'chi2', 'L1'], default="L1", \
                        help='Specifying the type of feature expert to be used')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    args = parser.parse_args()
    
    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset(args.dataset)
    
    num_inst, num_feat = X_pool.shape
    
    if not feat_names:
        feat_names = np.arange(num_feat)
    
    feature_frequency = np.diff(X_pool.tocsc().indptr)
    
    c0_frequency = np.diff(X_pool[np.nonzero(y_pool==0)[0]].tocsc().indptr)
    c1_frequency = np.diff(X_pool[np.nonzero(y_pool==1)[0]].tocsc().indptr)
        
    fe = feature_expert(X_pool, y_pool, metric=args.metric, smoothing=1e-6, C=args.c, pick_only_top=False)
    
    print '-' * 50
    
    print "FEATURE ANALYSIS"
    
    print '-' * 50
    
    print "Number of all features: %d" %(X_pool.shape[1])
    print "Number of non-zero features: %d" %(len(fe.feature_rank[0]) + len(fe.feature_rank[1]))
    print "# of class 0 features: %d" %(len(fe.feature_rank[0]))
    print "# of class 1 features: %d" %(len(fe.feature_rank[1]))
    
    # number of times a feature is the top feature in its document
    the_top = np.zeros(num_feat)
    
    for i in range(num_inst):
        mif = fe.most_informative_feature(X_pool[i], y_pool[i])
        if mif:
            the_top[mif] += 1 
    # print index, feature frequency, l1 weight, and # it is the top
    
    print
    
    print '-' * 50
    
    print "FEATURE IDS, NAMES, FREQUENCIES, SCORES, AND TOP COUNTS"
    
    print '-' * 50
    
    print "ID\tNAME\tFREQ\tC0_FREQ\tC1_FREQ\tSCORE\tTOP_COUNT"
    
    for f in range(num_feat):
        print "%d\t%s\t%d\t%d\t%d\t%0.4f\t%d" %(f, feat_names[f].encode('utf8'), \
                                                 feature_frequency[f], c0_frequency[f], \
                                                 c1_frequency[f], fe.feature_scores[f], the_top[f])
    
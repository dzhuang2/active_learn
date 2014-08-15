'''
Created on Aug 14, 2014

Prints information about:

What is the number of all features?
What is the number of features that have non-zero weight (for a logistic regression that is trained on the full training dataset)?
What is the class distribution of the features that have non-zero weight?

'''

import argparse

from feature_expert import feature_expert
from active_learn import load_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='imdb', nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    args = parser.parse_args()
    
    (X_pool, y_pool, X_test, y_test) = load_dataset(args.dataset)
        
    fe = feature_expert(X_pool, y_pool, metric="L1", smoothing=1e-6, C=args.c)
    
    print '-' * 50
    
    print "FEATURE ANALYSIS"
    
    print '-' * 50
    
    print "Number of all features: %d" %(X_pool.shape[1])
    print "Number of non-zero features: %d" %(len(fe.feature_rank[0]) + len(fe.feature_rank[1]))
    print "# of class 0 features: %d" %(len(fe.feature_rank[0]))
    print "# of class 1 features: %d" %(len(fe.feature_rank[1]))
    
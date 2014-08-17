'''
Created on Aug 14, 2014

Prints information about:

What is the performance of a classifier trained on the full training data?
What is the performance of a Feature Model that is trained using all features that have non-zero weight?
What is the performance of a Feature Model that is trained using top k features from each class?

'''

import argparse

from feature_expert import feature_expert
from active_learn import load_dataset, evaluate_model


from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from models import FeatureMNB

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default=['imdb'], nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    parser.add_argument('-k', type=int, default=10, help='number of features to use from each class')
    parser.add_argument('-smoothing', type=float, default=0, help='smoothing parameter for the feature MNB model')
   
    args = parser.parse_args()
    
    (X_pool, y_pool, X_test, y_test) = load_dataset(args.dataset)
    
    models = {'MultinomialNB(alpha=1)':MultinomialNB(alpha=1), \
              'LogisticRegression(C=1, penalty=\'l1\')':LogisticRegression(C=1, penalty='l1'), \
              'LogisticRegression(C=0.1, penalty=\'l1\')':LogisticRegression(C=0.1, penalty='l1')}
    
    aucs = {}
    
    for mk in models.keys():
        models[mk].fit(X_pool, y_pool)
        _, auc = evaluate_model(models[mk], X_test, y_test)
        aucs[mk] = auc
    
    fe = feature_expert(X_pool, y_pool, metric="L1", C=args.c)
    
    all_feature_model = FeatureMNB(fe.feature_rank[0], fe.feature_rank[1], fe.num_features, smoothing=args.smoothing)
    all_feature_model.update()
    
    _, all_auc = evaluate_model(all_feature_model, X_test, y_test)
    
    
    k_feature_model = FeatureMNB(fe.feature_rank[0][:args.k], fe.feature_rank[1][:args.k], fe.num_features, smoothing=args.smoothing)
    k_feature_model.update()
    
    _, k_auc = evaluate_model(k_feature_model, X_test, y_test)
    
    
    
    
    print '-' * 50
    
    print "DIFFICULTY ANALYSIS"
    
    print '-' * 50
    
    print "Instance Model AUCs"
    for mk in models.keys():
        print "%s \t: %0.4f" %(mk, aucs[mk])
    
    print
    print "Feature Model AUCs"
    
    print "Using all non_zero features \t: %0.4f" %all_auc
    
    
    print "Using top %d non_zero features from each class \t: %0.4f" % (args.k, k_auc)
    
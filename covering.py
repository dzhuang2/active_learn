'''
covering.py

This file contains functions that investigates how much each word 
covers the imdb corpus.
'''
from time import time
from feature_expert import feature_expert
from imdb import load_imdb, load_newsgroups
import numpy as np
import scipy.sparse as sp
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
import warnings
warnings.filterwarnings("ignore", category=sp.SparseEfficiencyWarning)

def covering(dataset='imdb', first='positive', agreement='any', metric='mutual_info', smoothing=1e-6, C=1):
    if first == 'positive':
        offset = 1
    else:
        offset = 0
    class_label = {0:'negative', 1:'positive'}
    vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
    
    if dataset == 'imdb':
        X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = load_imdb("./aclImdb", shuffle=True, vectorizer=vect)
    elif isinstance(dataset, tuple) and len(dataset) == 3 and dataset[0] == 'newsgroups':
        X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], shuffle=False, random_state=42, \
            vectorizer=vect)
    
    feature_names = vect.get_feature_names()
    fe = feature_expert(X_pool, y_pool, metric, smoothing, C)
        
    print 'class 0 features (ranked):'
    print ', '.join([str((f, feature_names[f])) for f in fe.class0_features_by_rank()])
    
    print 'class 1 features (ranked):'
    print ', '.join([str((f, feature_names[f])) for f in fe.class1_features_by_rank()])
    
    sample_pool = range(X_pool.shape[0])
    feature_list = list()
    X_csc = X_pool.tocsc()
    
    feature_num = 0

    while len(sample_pool) != 0:
        label = (feature_num + offset) % 2 # label for the document
        rank = feature_num / 2 # rank of the feature in the list
        feature_num += 1
        
        if rank < len(fe.feature_rank[label]):
            feature = fe.feature_rank[label][rank]
        else:
            print '*' * 50
            print ', '.join(['#'+str(doc) for doc in sample_pool]) + ' are uncovered'
            for doc in sample_pool:
                print '-' * 50
                print 'Document #%d:' % doc
                print '=' * 50
                print 'length = %d' % len(X_pool_docs[doc])
                print X_pool_docs[doc]
                print '=' * 50
                print X_pool[doc].indices
            break
            
        feature_name = feature_names[feature]
        docs_with_feature = X_csc.getcol(feature).indices

        docs_in_pool_with_feature = list(set(sample_pool).intersection(set(docs_with_feature)))
        if len(docs_in_pool_with_feature) == 0:
            continue
        else:
            num_docs_covered = len(docs_in_pool_with_feature)
            num_positive_docs = len(np.nonzero(y_pool[docs_in_pool_with_feature] == 1)[0])
            num_negative_docs = len(np.nonzero(y_pool[docs_in_pool_with_feature] == 0)[0])

            poolsize_before_removal = len(sample_pool)
            
            if agreement == 'agree':
                docs_with_label = np.nonzero(y_pool == label)[0]
                docs_to_remove = list(set(docs_in_pool_with_feature).intersection(set(docs_with_label)))
                sample_pool = list(set(sample_pool).difference(set(docs_to_remove)))
            else:
                sample_pool = list(set(sample_pool).difference(set(docs_in_pool_with_feature)))

            # pack the information into a dictionary for easy printing   
            result = dict()
            result['name'] = feature_name
            result['num'] = feature
            result['class'] = class_label[label]
            result['poolsize_before_removal'] = poolsize_before_removal
            result['num_docs_covered'] = num_docs_covered
            result['num_positive_docs'] = num_positive_docs
            result['num_negative_docs'] = num_negative_docs
            result['poolsize_after_removal'] = len(sample_pool)
            
            feature_list.append(result)

    return feature_list

def print_covering_details(result, filename='covering_full.txt'):
    print 'writing cover results to file \'%s\'' % filename
    with open(filename, 'w') as f:    
        for feature in result:
            f.write('-' * 50 + '\n')
            f.write('feature name: %s\n' % feature['name'].encode('utf8'))
            f.write('feature number: %d\n' % feature['num'])
            f.write('feature class: %s\n' % feature['class'])
            f.write('sample poolsize before: %d\n' % feature['poolsize_before_removal'])
            f.write('documents covered: %d, +: %d, -: %d\n' % \
                (feature['num_docs_covered'], feature['num_positive_docs'], feature['num_negative_docs']))
            f.write('sample poolsize after: %d\n' % feature['poolsize_after_removal'])

def print_summary(result, filename='covering_summary.txt'):
    print 'writing cover results to file \'%s\'' % filename
    with open(filename, 'w') as f: 
        for feature in result:
            f.write('-' * 50 + '\n')
            f.write('feature name: %s \t feature class: %s \t documents covered: %d\n' % \
                (feature['name'].encode('utf8'), feature['class'], feature['num_docs_covered']))
            f.write('postive documents covered: %d, negative documents covered: %d\n' % \
                (feature['num_positive_docs'], feature['num_negative_docs']))

if __name__ == '__main__':
    label = 'negative'
    agreement = 'agree' # agreement can be 'agree' or 'any'
    # metric = 'mutual_info' # metric should be either 'mutual_info' or 'L1'
    metric = 'L1'
    result = covering(dataset=('newsgroups', 'alt.atheism', 'talk.religion.misc'), \
                      first=label, metric=metric, agreement=agreement, C=0.1)
    # result = covering(dataset=('newsgroups', 'comp.graphics', 'comp.windows.x'), \
                      # first=label, metric=metric, agreement=agreement, C=0.1)
    # result = covering(dataset=('newsgroups', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware'), \
                      # first=label, metric=metric, agreement=agreement, C=0.1)
    # result = covering(dataset=('newsgroups', 'rec.sport.baseball', 'sci.crypt'), \
                      # first=label, metric=metric, agreement=agreement, C=0.1)
    summary_name = '-'.join([label, 'first', agreement, metric, 'summary.txt'])
    # print_summary(result, filename=summary_name)
    print_covering_details(result, filename=summary_name)


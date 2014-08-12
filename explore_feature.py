import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from feature_expert import feature_expert, print_all_features
from imdb import load_imdb, load_newsgroups
from sklearn import linear_model
from time import time
import sys
from sklearn.naive_bayes import MultinomialNB
from active_learn import evaluate_model, load_dataset
import argparse

class FeatureMNB(MultinomialNB):
    def __init__(self, class0_features, class1_features, num_feat, smoothing, class_prior = [0.5, 0.5], r=100.):
        self.class0_features = list(class0_features)
        self.class1_features = list(class1_features)
        self.smoothing = smoothing
        self.num_features = num_feat
        self.class_prior = class_prior
        self.r = r
    
        unlabeled_features = set(range(self.num_features))
        unlabeled_features.difference_update(self.class0_features)
        unlabeled_features.difference_update(self.class1_features)
        unlabeled_features = list(unlabeled_features)

        n0 = len(self.class0_features) # p
        n1 = len(self.class1_features) # n
        nu = len(unlabeled_features) # m-p-n

        #smoothing
        n0 += self.smoothing
        n1 += self.smoothing
        nu -= 2 * self.smoothing

        self.feature_log_prob_ = np.zeros(shape=(2,self.num_features))

        if self.class0_features != []:
            self.feature_log_prob_[0][self.class0_features] = np.log(1./(n0+n1)) # Equation 12
            self.feature_log_prob_[1][self.class0_features] = np.log(1./((n0+n1)*self.r)) # Equation 13
        elif self.class1_features != []:
            self.feature_log_prob_[1][self.class1_features] = np.log(1./(n0+n1)) # Equation 12
            self.feature_log_prob_[0][self.class1_features] = np.log(1./((n0+n1)*self.r)) # Equation 13
        
        #Equation 14
        self.feature_log_prob_[0][unlabeled_features] = np.log((n1*(1-1./self.r))/((n0+n1)*nu))
        self.feature_log_prob_[1][unlabeled_features] = np.log((n0*(1-1./self.r))/((n0+n1)*nu))

        self.class_log_prior_ = np.log(self.class_prior)
        self.classes_ = np.array([0, 1])

class alt_L1_feature_expert(feature_expert):
    def __init__(self, X, y, metric, smoothing=1e-6, C=0.1):
        self.sample_size, self.num_features = X.shape
        self.metric = metric
        self.smoothing = smoothing
        
        print '-' * 50
        print 'Starting Feature Expert Training ...'
        start = time()
        
        self.feature_count = self.count_features(X, y)
        clf_l1 = linear_model.LogisticRegression(C=C, penalty='l1')
        clf_l1.fit(X, y)
        self.L1_weights = clf_l1.coef_[0]
        
        if metric == 'weight':
            class0_features = np.nonzero(self.L1_weights < 0)[0]
            class1_features = np.nonzero(self.L1_weights > 0)[0]
            class0_features_ranked = class0_features[np.argsort(self.L1_weights[class0_features])]
            class1_features_ranked = class1_features[np.argsort(self.L1_weights[class1_features])[::-1]]
            self.feature_rank = (class0_features_ranked, class1_features_ranked)
        elif metric == 'non_zero':
            non_zero_features = np.nonzero(self.L1_weights != 0)[0]
            non_zero_index_sorted = np.argsort(np.absolute(self.L1_weights[non_zero_features]))[::-1]
            feature_rank = non_zero_features[non_zero_index_sorted]
            self.feature_rank = self.classify_features(feature_rank)
        else:
            raise ValueError('metric must be one of the following: \'weight\', \'non_zero\'')
        
        print 'Feature Expert has deemed %d words to be of class 0' % len(self.feature_rank[0])        
        print 'Feature Expert has deemed %d words to be of class 1' % len(self.feature_rank[1])
        
        print 'Feature Expert trained in %0.2fs' % (time() - start)

if __name__ == '__main__':
    '''
    C = 0.1, min_df is 5 and type is 'weight' by default
    python test1.py -cat alt.atheism talk.religion.misc -c 0.1 -d 5
    python test1.py -cat comp.graphics comp.windows.x
    python test1.py -cat comp.os.ms-windows.misc comp.sys.ibm.pc.hardware
    python test1.py -cat rec.sport.baseball sci.crypt
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-cat', default=['alt.atheism', 'talk.religion.misc'], nargs=2, \
                        help='2 class labels from the 20newsgroup dataset')
    parser.add_argument('-dataset', default='SRAA', help='dataset')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    parser.add_argument('-d', type=int, default=5, help='Min_df for CountVectorizer')
    parser.add_argument('-type', default='weight', choices=['weight', 'non_zero'], help='Type of metric used to' + \
                        'partition the features into the two classes')
    args = parser.parse_args()
    
    vect = CountVectorizer(min_df=args.d, max_df=1.0, binary=True, ngram_range=(1, 1))
    
    X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = load_imdb(path='./aclImdb', shuffle=True, vectorizer=vect)
    feature_names = np.array(vect.get_feature_names())
    
    # X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = \
        # load_newsgroups(args.cat[0], args.cat[1], shuffle=True, random_state=42, \
            # remove=('headers', 'footers'), vectorizer=vect)
    # feature_names = vect.get_feature_names()
    
    # X_pool, y_pool, X_test, y_test = load_dataset(args.dataset)
    # X_pool_docs = pickle.load(open('SRAA_X_train_corpus.pickle', 'rb'))
    # X_test_docs = pickle.load(open('SRAA_X_test_corpus.pickle', 'rb'))
    
    # feature_names = pickle.load(open('SRAA_feature_names.pickle', 'rb'))
    
    fe = alt_L1_feature_expert(X_pool, y_pool, args.type, smoothing=1e-6, C=args.c)
    
    print 'class 0 features (ranked):'
    print ', '.join([str((f, feature_names[f], fe.L1_weights[f])) for f in fe.class0_features_by_rank()])
    print '-' * 50
    
    print 'class 1 features (ranked):'
    print ', '.join([str((f, feature_names[f], fe.L1_weights[f])) for f in fe.class1_features_by_rank()])
    print '-' * 50
    
    # doc_ids = np.random.permutation(np.arange(X_pool.shape[0]))
    # top_n = 20
    
    # print '\n'
    # print '=' * 50
    
    # for doc in doc_ids:
        # print_all_features(feature_names, fe, top_n, doc, X_pool, y_pool, X_pool_docs)
        
        # print '=' * 50
        # ch = raw_input('Display the next document? Press Enter to continue or type \'n\' to exit...  ')
        
        # if ch == 'n':
            # break
    
    feature_model = FeatureMNB(fe.class0_features_by_rank(), fe.class1_features_by_rank(), \
        num_feat=X_pool.shape[1], smoothing=1e-6, class_prior = [0.5, 0.5], r=100.)
    
    print 'Feature Model(MNB): accu = %f, auc = %f' % evaluate_model(feature_model, X_test, y_test)
    
    logit = linear_model.LogisticRegression(C=args.c, penalty='l1')
    logit.fit(X_pool, y_pool)
    
    print 'Feature Model(LogisticRegression): accu = %f, auc = %f' % evaluate_model(logit, X_test, y_test)
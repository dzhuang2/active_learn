'''
selection_strategies.py

This file contains the following:
    RandomBootstrap: 
    RandomStrategy: pick the next document randomly from pool
    UNCSampling: pick the next document based on uncertainty
    DisagreementStrategy: pick the next document based on how much the instance model and feature model disagree
    CoveringStrategy: pick the next document that does NOT have any of its features annotated
    CoverThenUncertainty: cover x% of the documents then perform uncertainty sampling using the pooling model
    CoveringFewest: pick the next document with the among the fewest number of features annotated
    CheatingApproach: pick the next document using feature expert's rankings
'''
import sys
from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from imdb import load_imdb
from feature_expert import print_all_features

def RandomBootstrap(X_pool, y_pool, size, balance, seed=0):
    '''
    Assume the task is binary classification
    '''
    print '-' * 50
    print 'Starting bootstrap...'
    print 'Initial training set size = %d' % size
    start = time()
    
    np.random.seed(seed)
    poolsize = y_pool.shape[0]
    
    pool_set = np.arange(poolsize)
    
    if balance: # select 1/2 * size from each class
        class0_size = size / 2
        class1_size = size - class0_size
        class0_indices = np.nonzero(y_pool == 0)[0]
        class1_indices = np.nonzero(y_pool == 1)[0]
        
        class0_docs = np.random.permutation(class0_indices)[:class0_size]
        class1_docs = np.random.permutation(class1_indices)[:class1_size]
        
        training_set = np.hstack((class0_docs, class1_docs))
        
    else: # otherwise, pick 'size' documents randomly
        training_set = np.random.permutation(pool_set)[:size]
    
    pool_set = np.setdiff1d(pool_set, training_set)
    
    print 'bootstraping took %0.2fs.' % (time() - start)
    
    return (training_set.tolist(), pool_set.tolist())

class RandomStrategy(object):
    def __init__(self, seed=0):
        self.rgen = np.random
        self.rgen.seed(seed)
        
    def choice(self, X, pool):
        choice = self.rgen.randint(len(pool))
        return pool[choice]

class UNCSampling(object):
    '''
    This class performs uncertainty sampling based on the model
    
    The model used is either instance_model, feature_model or pooling_model
    '''
    def __init__(self, model, feature_expert, y, Debug=False):
        self.model = model
        self.Debug = Debug
        
        if self.Debug:
            self.feature_expert = feature_expert
            self.y = y
            self.top_n, self.X_pool, self.y_pool, self.X_pool_docs, self.feature_names = \
                load_Debug_data()
    
    def choice(self, X, pool):
        y_probas = self.model.predict_proba(X[pool])
        doc_id = pool[np.argsort(np.absolute(y_probas[:, 1] - 0.5))[0]]
        
        if self.Debug:
            print '\n'
            print '=' * 50
            # print 'Feature model thus far:'
            # print '*' * 50
            # print 'Negative features (class 0):'
            # print ', '.join(self.feature_names[self.model.class0_features])
            # print 'Positive features (class 1):'
            # print ', '.join(self.feature_names[self.model.class1_features])
            # print '=' * 50
            print_all_features(self.feature_names, self.feature_expert, self.top_n, doc_id, self.X_pool, self.y_pool, self.X_pool_docs)
            doc_prob = self.model.predict_proba(self.X_pool[doc_id])
            print 'feature model predict_probability class0 = %0.5f, class1 = %0.5f' % (doc_prob[0, 0], doc_prob[0, 1])
            
            feature = self.feature_expert.most_informative_feature(self.X_pool[doc_id], self.y_pool[doc_id])
            print 'feature to be added to the model = (%d, %s)' % (feature, self.feature_names[feature])
            print 'label to be added to the model = %d' % self.y_pool[doc_id]
            print
            
            print 'making sure that X_pool and X are indeed the same:'
            print 'label according to y: %d' % self.y[doc_id]
            x_feature = self.feature_expert.most_informative_feature(X[doc_id], self.y[doc_id])
            print 'feature according to X: (%d, %s)' % (x_feature, self.feature_names[x_feature])
            
            ch = raw_input('Press Enter to continue...  ')
            print 
            
            if ch == 'n':
                sys.exit(1)
        
        return doc_id
        
class DisagreementStrategy(object):
    def __init__(self, instance_model, feature_model, feature_expert, y, metric, seed=50, Debug=False):
        self.instance_model = instance_model
        self.feature_model = feature_model
        self.metric = metric
        self.Debug = Debug
        self.rgen = np.random
        self.rgen.seed(0)
        
        if self.Debug:
            self.feature_expert = feature_expert
            self.y = y
            self.top_n, self.X_pool, self.y_pool, self.X_pool_docs, self.feature_names = \
                load_Debug_data()
        
    def choice(self, X, pool):
        if self.metric == 'KLD':
            return self.KLD(X, pool)
        elif self.metric == 'vote':
            return self.vote(X, pool)
        elif self.metric == 'euclidean':
            return self.euclidean(X, pool)
        else:
            raise ValueError('metric must be one of the following: \'KLD\'!')

    def euclidean(self, X, pool):
        y_IM_probas = self.instance_model.predict_proba(X[pool])
        y_FM_probas = self.feature_model.predict_proba(X[pool])

        dist = np.sum(np.multiply(y_IM_probas - y_FM_probas, y_IM_probas - y_FM_probas), axis=1)
        # select the document with the largest euclidean distance
        doc_id = np.array(pool)[np.argsort(dist)[-1]]

        if self.Debug:
            print '\n'
            print '=' * 50
            print 'Feature model thus far:'
            print '*' * 50
            print 'Negative features (class 0):'
            print ', '.join(self.feature_names[self.feature_model.class0_features])
            print 'Positive features (class 1):'
            print ', '.join(self.feature_names[self.feature_model.class1_features])
            print '=' * 50
            print_all_features(self.feature_names, self.feature_expert, self.top_n, doc_id, self.X_pool, self.y_pool, self.X_pool_docs)
            
            IM_prob = self.instance_model.predict_proba(self.X_pool[doc_id])
            print 'instance model predict_probability: class0 = %0.5f, class1 = %0.5f' % (IM_prob[0, 0], IM_prob[0, 1])
            
            FM_prob = self.feature_model.predict_proba(self.X_pool[doc_id])
            print 'feature model predict_probability:  class0 = %0.5f, class1 = %0.5f' % (FM_prob[0, 0], FM_prob[0, 1])

            sorted_dist = np.argsort(dist)
            print 'top 10 Euclidean Distances:'
            print dist[sorted_dist[-10:]]

            print sorted_dist[:10]
            for i in range(1, self.top_n + 1):
                print 'Rank %d: doc#%d, distance=%10.5f' % (i, pool[sorted_dist[-i]], dist[sorted_dist[-i]])
            
            print 'this doc\'s distance = ', dist[sorted_dist[-1]]

            ch = raw_input('Press Enter to continue...  ')
            print 
            
            if ch == 'n':
                sys.exit(1)
            
        return doc_id

    def vote(self, X, pool):
        y_IM_probas = self.instance_model.predict_proba(X[pool])
        y_FM_probas = self.feature_model.predict_proba(X[pool])

        y_IM_probas[y_IM_probas >= 0.5] = 1
        y_IM_probas[y_IM_probas < 0.5] = 0

        y_FM_probas[y_FM_probas >= 0.5] = 1
        y_FM_probas[y_FM_probas < 0.5] = 0

        vote = y_IM_probas + y_FM_probas
        docs = vote[:,0] == 1 # these docs would have conflicting votes from IM and PM
        doc_ids = np.array(pool)[docs].tolist()
        
        choice = self.rgen.randint(len(doc_ids)) # choose a random document from the list of doc_ids

        print 'number of documents with conflicting votes: %d' % len(doc_ids)

        return doc_ids[choice]

    def KLD(self, X, pool):
        '''
        Compute average KL Divergence between instance and feature model,
        the larger the value, the more KLD says that they disagree
        
        avg_KLD(IM, FM) = (KLD(IM, FM) + KLD(FM, IM)) / 2
        '''
        y_IM_probas = self.instance_model.predict_proba(X[pool])
        y_FM_probas = self.feature_model.predict_proba(X[pool])
        
        log_ratio = np.log(y_IM_probas) - np.log(y_FM_probas)
        KLD_IM_FM = np.sum(y_IM_probas *  log_ratio, 1)
        KLD_FM_IM = np.sum(y_FM_probas * -log_ratio, 1)
        KLD = (KLD_IM_FM + KLD_FM_IM) / 2
        
        num = np.argsort(KLD)[-1]
        doc_id = pool[num]
        
        if self.Debug:
            print '\n'
            print '=' * 50
            print 'Feature model thus far:'
            print '*' * 50
            print 'Negative features (class 0):'
            print ', '.join(self.feature_names[self.feature_model.class0_features])
            print 'Positive features (class 1):'
            print ', '.join(self.feature_names[self.feature_model.class1_features])
            print '=' * 50
            print_all_features(self.feature_names, self.feature_expert, self.top_n, doc_id, self.X_pool, self.y_pool, self.X_pool_docs)
            
            IM_prob = self.instance_model.predict_proba(self.X_pool[doc_id])
            print 'instance model predict_probability: class0 = %0.5f, class1 = %0.5f' % (IM_prob[0, 0], IM_prob[0, 1])
            
            FM_prob = self.feature_model.predict_proba(self.X_pool[doc_id])
            print 'feature model predict_probability:  class0 = %0.5f, class1 = %0.5f' % (FM_prob[0, 0], FM_prob[0, 1])

            print 'top 10 KLDs:'
            sorted_KLD = np.argsort(KLD)
            for i in range(1, self.top_n + 1):
                print 'Rank %d: doc#%d, KLD=%10.5f' % (i, pool[sorted_KLD[-i]], KLD[sorted_KLD[-i]])
            
            print 'this doc\'s KLD = ', KLD[num]
            
            feature = self.feature_expert.most_informative_feature(self.X_pool[doc_id], self.y_pool[doc_id])
            print 'feature to be added to the model = (%d, %s)' % (feature, self.feature_names[feature])
            print 'label to be added to the model = %d' % self.y_pool[doc_id]
            print
            
            print 'making sure that X_pool and X are indeed the same:'
            print 'label according to y: %d' % self.y[doc_id]
            x_feature = self.feature_expert.most_informative_feature(X[doc_id], self.y[doc_id])
            print 'feature according to X: (%d, %s)' % (x_feature, self.feature_names[x_feature])
            
            ch = raw_input('Press Enter to continue...  ')
            print 
            
            if ch == 'n':
                sys.exit(1)
        
        return doc_id

class CoveringStrategy(object):
    def __init__(self, feature_expert, num_samples, y, type='unknown', seed=0, Debug=False):
        self.docs_feature_count = np.zeros(num_samples) # a list of documents with annotated features counted
        self.rgen = np.random
        self.rgen.seed(seed)
        self.annotated_features = list() # for Debugging mostly
        self.type = type
        self.Debug = Debug
        self.DebugStr = self.header()

        if self.Debug:
            self.feature_expert = feature_expert
            self.y = y
            self.top_n, self.X_pool, self.y_pool, self.X_pool_docs, self.feature_names = \
                load_Debug_data()
    
    def header(self):
        header = ['InPool', 'Total', 'Before', 'After', 'Removed', 'Feature#', 'Word']
        return ''.join([s.ljust(10) for s in header]) + '\n'
    
    def addDebugInfo(self, *args):
        result = []
        for val in args:
            if isinstance(val, basestring):
                result.append(val.encode('utf8'))
            elif isinstance(val, int):
                result.append('{:<5d}'.format(val))
        self.DebugStr += '     '.join(result) + '\n'
    
    def printDebugInfoToFile(self, filename='covering_debug.txt'):
        with open(filename, 'w') as f:
            f.write(self.DebugStr)
    
    def choice(self, X, pool):
        if self.type == 'unknown':
            min = 0
        elif self.type == 'fewest':
            min = np.min(self.docs_feature_count[pool])
        else:
            raise ValueError('Covering stratgy must be either \'noknown\' or \'fewest\'')
        
        min_indices = np.nonzero(self.docs_feature_count == min)[0]
        sampling_pool = list(set(pool).intersection(set(min_indices)))
        
        if len(sampling_pool) == 0:
            doc_id = None
        else:
            doc_id = self.rgen.permutation(sampling_pool)[0]
        
#        if self.Debug and doc_id != None:
#            print 'min number of features: %d' % min
#            print 'num of documents in the sampling_pool: %d' % len(sampling_pool)
#            
#            print '%d features annotated so far:' % len(self.annotated_features)
#            print ', '.join([str((f, self.feature_names[f])) for f in self.annotated_features])
#            
#            print_all_features(self.feature_names, self.feature_expert, self.top_n, doc_id, self.X_pool, self.y_pool, self.X_pool_docs)
#            
#            feature = self.feature_expert.most_informative_feature(self.X_pool[doc_id], self.y_pool[doc_id])
#            print 'feature to be added to the model = (%d, %s)' % (feature, self.feature_names[feature])
#            print 'label to be added to the model = %d' % self.y_pool[doc_id]
#            print
#            
#            print 'making sure that X_pool and X are indeed the same:'
#            print 'label according to y: %d' % self.y[doc_id]
#            x_feature = self.feature_expert.most_informative_feature(X[doc_id], self.y[doc_id])
#            print 'feature according to X: (%d, %s)' % (x_feature, self.feature_names[x_feature])
#            
#            ch = raw_input('Press Enter to continue...  ')
#            print '-' * 50
#            
#            if ch == 'n':
#                sys.exit(1)
        
        return doc_id
    
    def update(self, X, feature, docid):
        # if top feature is None, increment the count for the document to remove it from sampling pool
        if feature == None:
            self.docs_feature_count[docid] += 1
            return
            
        # if feature is not None
        X_csc = X.tocsc()
        docs_with_features = X_csc.getcol(feature).indices

        if self.type == 'unknown':
            min = 0
        elif self.type == 'fewest':
            min = np.min(self.docs_feature_count)
        
        if self.Debug:
            docs_uncovered_before = np.nonzero(self.docs_feature_count == min)[0]
            print 'feature to be removed: (%d, %s)' % (feature, self.feature_names[feature])
            print 'total number of documents with this feature = %d' % \
                    len(docs_with_features)
            print 'total number of documents with this feature in the pool = %d' % \
                    len(list(set(docs_uncovered_before).intersection(docs_with_features)))
            print 'number of documents in pool with no known features (before) = %d' % \
                     docs_uncovered_before.shape[0]
            Feature_num = feature
            Word = self.feature_names[feature]
            Total = len(docs_with_features)
            InPool = len(list(set(docs_uncovered_before).intersection(docs_with_features)))
            Before = docs_uncovered_before.shape[0]
            
        self.docs_feature_count[docs_with_features] += 1
        
        if self.Debug:
            docs_uncovered_after = np.nonzero(self.docs_feature_count == min)[0]
            # print 'number of documents in pool with no known features (after) = %d' % \
                    # docs_uncovered_after.shape[0]
            # print 'number of documents removed from pool = %d' % \
                    # (docs_uncovered_after.shape[0] - docs_uncovered_before.shape[0])
            After = docs_uncovered_after.shape[0]
            Removed = docs_uncovered_before.shape[0] - docs_uncovered_after.shape[0]
            line = [InPool, Total, Before, After, Removed, Feature_num, Word]
            self.addDebugInfo(*line)
        
        if feature not in self.annotated_features:
            self.annotated_features.append(feature)

class CheatingApproach(object):
    def __init__(self, feature_expert, num_samples, y, seed=0, Debug=False):
        self.rgen = np.random
        self.rgen.seed(seed)
        self.annotated_features = list()
        self.docs_feature_count=np.zeros(num_samples)
        self.Debug = Debug
        self.y = y
        self.feature_expert = feature_expert
        
        if self.Debug:
            self.top_n, self.X_pool, self.y_pool, self.X_pool_docs, self.feature_names = \
                load_Debug_data()
    
    def choice(self, X, pool):
        # if len(annotated_features) is even, choose a positive document
        # if len(annotated_features) is odd, choose a negative document
        
        label = len(self.annotated_features) % 2 # label for the document
        rank = len(self.annotated_features) / 2 # rank of the feature in the list
        feature = self.feature_expert.feature_rank[label][rank]
        
        # find all documents with next feature present
        X_csc = X.tocsc()
        docs_with_feature = X_csc.getcol(feature).indices
        
        # find the docs with no annotated features
        doc_with_no_annotated_features = np.nonzero(self.docs_feature_count == 0)[0]
        
        # Find documents without any annotated features but has the next feature
        potential_docs = set(docs_with_feature).intersection(set(doc_with_no_annotated_features))
        
        # Find indices of all labels that is the current label
        correct_label_indices = np.nonzero(self.y == label)[0]
        
        # Find the intersection between the result from above and the pool
        sampling_pool = list((set(pool).intersection(potential_docs)).intersection(correct_label_indices))
        
        if len(sampling_pool) == 0:
            doc_id = None
        else:
            doc_id = self.rgen.permutation(sampling_pool)[0]
        
        if self.Debug and doc_id != None:
            print 'size of overall pool: %d' % len(pool)
            print 'number of samples with feature present: %d' % len(docs_with_feature)
            print 'number of samples with no annotated_features: %d' % len(doc_with_no_annotated_features)
            print 'number of samples with label=%d: %d' % (label, correct_label_indices.shape[0])
            print 'size of the sampling pool: %d' % len(sampling_pool)
            
            print 'Annotated Features(%d): ' % len(self.annotated_features)
            print ', '.join([str((f, self.feature_names[f])) for f in self.annotated_features])
            print 'Cheating Approach: rank = %d, feature# = %d, feature name = %s' % (rank, feature, self.feature_names[feature])
            
            print_all_features(self.feature_names, self.feature_expert, self.top_n, doc_id, self.X_pool, self.y_pool, self.X_pool_docs)
            
            feature = self.feature_expert.most_informative_feature(self.X_pool[doc_id], self.y_pool[doc_id])
            print 'feature to be added to the model = (%d, %s)' % (feature, self.feature_names[feature])
            print 'label to be added to the model = %d' % self.y_pool[doc_id]
            print
            
            print 'making sure that X_pool and X are indeed the same:'
            print 'label according to y: %d' % self.y[doc_id]
            x_feature = self.feature_expert.most_informative_feature(X[doc_id], self.y[doc_id])
            print 'feature according to X: (%d, %s)' % (x_feature, self.feature_names[x_feature])
            
            ch = raw_input('Press Enter to continue...  ')
            print '-' * 50
            
            if ch == 'n':
                sys.exit(1)
        
        return doc_id
        
    def update(self, X, feature, label):
        if feature not in self.annotated_features:
            X_csc = X.tocsc()
            docs_with_features = X_csc.getcol(feature).indices
            docs_with_label = np.nonzero(self.y == label)[0]
            
            docs_to_be_removed = list(set(docs_with_features).intersection(docs_with_label))
            
            if self.Debug:
                print 'feature to be removed: (%d, %s)' % (feature, self.feature_names[feature])
                print 'label to be added to the model = %d' % label
                print 'number of documents in pool with no known features = %d' % \
                        np.nonzero(self.docs_feature_count == 0)[0].shape[0]
                print 'number of documents to be removed from pool = %d' % len(docs_to_be_removed)
            
            # remove documents with annotated feature via set arithmetic
            self.annotated_features.append(feature)
            self.docs_feature_count[docs_to_be_removed] += 1

class CoveringThenDisagreement(object):
    def __init__(self, feature_expert, instance_model, feature_model, num_samples, percentage, y, \
                 type='unknown', metric='KLD', seed=0, Debug=False):
        self.covering = CoveringStrategy(feature_expert, num_samples, y, type, seed, Debug)
        self.KLD = DisagreementStrategy(instance_model, feature_model, feature_expert, y, metric, Debug)
        self.phase = 'covering'
        self.min_docs_covered = num_samples * percentage
        self.transition = None
        
    def choice(self, X, num, pool):
        if self.phase == 'covering':
            doc_id = self.covering.choice(X, pool)
            docs_covered = np.nonzero(self.covering.docs_feature_count > 0)[0].shape[0]
            if doc_id == None or docs_covered > self.min_docs_covered:
                self.phase = 'disagreement'
                self.transition = num
                print 'covering transition happens at sample #%d' % num
        
        if self.phase == 'disagreement':
            doc_id = self.KLD.choice(X, pool)
        
        return doc_id

class CoverThenUncertainty(object):
    def __init__(self, feature_expert, pooling_model, num_samples, percentage, y, \
                 type='unknown', seed=0, Debug=False):
        self.covering = CoveringStrategy(feature_expert, num_samples, y, type, seed, Debug)
        self.uncertainty = UNCSampling(pooling_model, feature_expert, y, Debug)
        self.phase = 'covering'
        self.min_docs_covered = num_samples * percentage
        self.transition = None
        
    def choice(self, X, num, pool):
        if self.phase == 'covering':
            doc_id = self.covering.choice(X, pool)
            docs_covered = np.nonzero(self.covering.docs_feature_count > 0)[0].shape[0]
            if doc_id == None or docs_covered > self.min_docs_covered:
                self.phase = 'uncertainty'
                self.transition = num
                print 'covering transition happens at sample #%d' % num
        
        if self.phase == 'uncertainty':
            doc_id = self.uncertainty.choice(X, pool)
        
        return doc_id
        
def load_Debug_data(top_n=10, min_df=5, max_df=1.0, binary=True, ngram_range=(1,1), \
                    shuffle=False, path='./aclImdb'):
    vect = CountVectorizer(min_df=min_df, max_df=max_df, binary=binary, ngram_range=ngram_range)
    print '=' * 50
    X_pool, y_pool, _, _, X_pool_docs, _ = load_imdb(path, shuffle=shuffle, vectorizer=vect)
    feature_names = np.array(vect.get_feature_names())
    return (top_n, X_pool, y_pool, X_pool_docs, feature_names)

from sklearn.naive_bayes import MultinomialNB
from models import FeatureMNBUniform, FeatureMNBWeighted, PoolingMNB
from sklearn import metrics

class OptimizeAUC(object):
    '''
    This class chooses the instance that is expected to lead to the maximum achievable AUC
    
    '''
    def __init__(self, X_test, y_test, feature_expert, optimize="P", seed=0, sub_pool = 100, Debug=False):
        self.X_test = X_test
        self.y_test = y_test
        self.feature_expert = feature_expert
        self.optimize=optimize        
        self.rgen = np.random
        self.rgen.seed(seed)
        self.sub_pool = sub_pool
        self.Debug = Debug
        
    
    def choice(self, X, y, pool, train_indices, current_feature_model):
        
        rand_indices = self.rgen.permutation(len(pool))
        candidates = [pool[i] for i in rand_indices[:self.sub_pool]]
        
        aucs = []
        
        for doc in candidates:
            new_train_indices = list(train_indices)
            new_train_indices.append(doc)
            
            # train an instance model
            instance_model = MultinomialNB(alpha=1.)
            instance_model.fit(X[new_train_indices], y[new_train_indices])
            
            # train a feature model
            
            feature_model = None
            if isinstance(current_feature_model, FeatureMNBUniform):
                feature_model = FeatureMNBUniform(current_feature_model.class0_feats, current_feature_model.class1_feats, self.feature_expert.num_features, 0)
            elif isinstance(current_feature_model, FeatureMNBWeighted):
                feature_model = FeatureMNBWeighted(num_feat = self.feature_expert.num_features, feat_count = current_feature_model.feature_count_, alpha = current_feature_model.alpha)
            else:
                raise ValueError('Feature model type: \'%s\' unknown!' % current_feature_model.__class__.__name__)
            
            top_feat = self.feature_expert.most_informative_feature(X[doc], y[doc])
            
            if top_feat:
                feature_model.fit(top_feat, y[doc]) # fit also calls update; so there is no need to update again
            else:
                feature_model.update()
            
            # pooling model
            
            pooling_model = PoolingMNB()
            pooling_model.fit(instance_model, feature_model, weights=[0.5, 0.5])
            
            # evaluate
            
            opt_model = None
            
            if self.optimize == "P":
                opt_model = pooling_model
            elif self.optimize == "I":
                opt_model = instance_model
            elif self.optimize == "F":
                opt_model = feature_model
            else:
                raise ValueError('Optimization Model: \'%s\' invalid!' % self.optimize)
            
            y_probas = opt_model.predict_proba(self.X_test)
            
            auc = metrics.roc_auc_score(self.y_test, y_probas[:, 1])
            aucs.append(auc)
                
        doc_id = candidates[np.argsort(aucs)[-1]]
        
        return doc_id
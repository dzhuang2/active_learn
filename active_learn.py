from time import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from imdb import load_imdb, load_newsgroups, load_nova
from models import FeatureMNBUniform, FeatureMNBWeighted, PoolingMNB, ReasoningMNB
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from feature_expert import feature_expert
from selection_strategies import RandomBootstrap, RandomStrategy, UNCSampling, DisagreementStrategy
from selection_strategies import CoveringStrategy, CheatingApproach, CoveringThenDisagreement, \
CoverThenUncertainty, ReasoningThenFeatureCertainty, CoverThenFeatureCertainty, UNCForInsufficientReason, \
UNCWithNoConflict, UNCPreferNoConflict, UNCPreferConflict
from selection_strategies import OptimizeAUC

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')

def load_data(pool_filename='./aclImdb/imdb-binary-pool-mindf5-ng11.bak', \
              test_filename='./aclImdb/imdb-binary-test-mindf5-ng11.bak', \
              n_features=27272):
     
    print '-' * 50
    print "Loading the data..."
    t0 = time()

    X_pool, y_pool = load_svmlight_file(pool_filename, n_features)
    X_test, y_test = load_svmlight_file(test_filename, n_features)

    duration = time() - t0

    print "Loading took %0.2fs." % duration
    
    return (X_pool, y_pool, X_test, y_test)

def learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, feature_expert, \
          selection_strategy, disagree_strat, coverage, budget, instance_model, feature_model, \
          pooling_model, reasoning_model, rmw_n, rmw_a, seed=0, Debug=False, \
          reasoning_strategy='random', switch=40):
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    instance_model_scores = {'auc':[], 'accu':[]}
    feature_model_scores = {'auc':[], 'accu':[]}
    pooling_model_scores = {'auc':[], 'accu':[]}
    reasoning_model_scores = {'auc':[], 'accu':[]}
        
    discovered_feature_counts = {'class0':[], 'class1': []}
    num_docs_covered = []    
    covered_docs = set()    
    X_pool_csc = X_pool.tocsc()
    
    num_samples = len(pool_set) + len(training_set)
    
    num_feat = X_pool.shape[1]
    
    num_a_feat_chosen = np.zeros(num_feat)
    
    discovered_features = set()
    
    discovered_class0_features = set()
    
    discovered_class1_features = set()
    
    np.random.seed(seed)
           
    if selection_strategy == 'random':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'uncertaintyIM':
        doc_pick_model = UNCSampling(instance_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'uncertaintyFM':
        doc_pick_model = UNCSampling(feature_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'uncertaintyPM':
        doc_pick_model = UNCSampling(pooling_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'uncertaintyRM':
        doc_pick_model = UNCSampling(reasoning_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'disagreement':
        doc_pick_model = DisagreementStrategy(instance_model, feature_model, \
            feature_expert, y_pool, disagree_strat, Debug=Debug)
    elif selection_strategy == 'covering':
        doc_pick_model = CoveringStrategy(feature_expert, num_samples, y_pool, \
            type='unknown', seed=seed, Debug=Debug)
    elif selection_strategy == 'covering_fewest':
        doc_pick_model = CoveringStrategy(feature_expert, num_samples, y_pool, \
            type='fewest', seed=seed, Debug=Debug)
    elif selection_strategy == 'cheating':
        doc_pick_model = CheatingApproach(feature_expert, num_samples, y_pool, \
            seed=seed, Debug=Debug)
    elif selection_strategy == 'cover_then_disagree':
        doc_pick_model = CoveringThenDisagreement(feature_expert, instance_model, \
            feature_model, num_samples, percentage=coverage, y=y_pool, type='unknown', \
            metric=disagree_strat, seed=seed, Debug=Debug)
    elif selection_strategy == 'cover_then_uncertaintyPM':
        doc_pick_model = CoverThenUncertainty(feature_expert, pooling_model, \
            num_samples, percentage=coverage, y=y_pool, type='unknown', \
            seed=seed, Debug=Debug)
    elif selection_strategy == 'cover_then_uncertaintyRM':
        doc_pick_model = CoverThenUncertainty(feature_expert, reasoning_model, \
            num_samples, percentage=coverage, y=y_pool, type='unknown', \
            seed=seed, Debug=Debug)
    elif selection_strategy == 'cover_then_featureCertainty':
        doc_pick_model = CoverThenFeatureCertainty(feature_expert, feature_model, \
            num_samples, percentage=coverage, y=y_pool, type='unknown', \
            seed=seed, Debug=Debug)
    elif selection_strategy == "optaucP":
        doc_pick_model = OptimizeAUC(X_test, y_test, feature_expert, \
            optimize="P", seed=seed, Debug=Debug)
    elif selection_strategy == "optaucI":
        doc_pick_model = OptimizeAUC(X_test, y_test, feature_expert, \
            optimize="I", seed=seed, Debug=Debug)
    elif selection_strategy == "optaucF":
        doc_pick_model = OptimizeAUC(X_test, y_test, feature_expert, \
            optimize="F", seed=seed, Debug=Debug)
    elif selection_strategy == "optaucR":
        doc_pick_model = OptimizeAUC(X_test, y_test, feature_expert, \
            optimize="R", seed=seed, Debug=Debug)  
    elif selection_strategy == 'reasoning_then_featureCertainty':
        doc_pick_model = ReasoningThenFeatureCertainty(feature_expert, instance_model, \
            feature_model, switch=switch, reasoning_strategy=reasoning_strategy, y=y_pool, type='unknown', \
            seed=seed, Debug=Debug)
    elif selection_strategy == "unc_insuff_R":
        doc_pick_model = UNCForInsufficientReason(reasoning_model)
    elif selection_strategy == "unc_no_conflict_R":
        doc_pick_model = UNCWithNoConflict(reasoning_model)
    elif selection_strategy == "unc_prefer_no_conflict_R":
        doc_pick_model = UNCPreferNoConflict(reasoning_model)
    elif selection_strategy == "unc_prefer_conflict_R":
        doc_pick_model = UNCPreferConflict(reasoning_model)
    else:
        raise ValueError('Selection strategy: \'%s\' invalid!' % selection_strategy)
    
    bootstrap_size = len(training_set)

    training_set_empty = (bootstrap_size == 0)
    
    if not training_set_empty:
        X_train = X_pool[training_set]
        y_train = y_pool[training_set]
        
        # Train all three models using the training set data
        instance_model.fit(X_train, y_train) # train instance_model
        
        for doc in training_set:
            #feature = feature_expert.most_informative_feature(X_pool[doc], y_pool[doc])
            feature = feature_expert.any_informative_feature(X_pool[doc], y_pool[doc])
            
            if feature:
                feature_model.fit(feature, y_pool[doc]) # train feature_model one by one
                discovered_features.add(feature)                
                if y_pool[doc] == 0:
                    discovered_class0_features.add(feature)
                else:
                    discovered_class1_features.add(feature)
                    
            # Reasoning model
            reasoning_model.partial_fit(X_pool[doc], y_pool[doc], feature, rmw_n, rmw_a) # train feature_model one by one
           
            # docs covered            
            if feature:
                f_covered_docs = X_pool_csc[:, feature].indices
                covered_docs.update(f_covered_docs)
            
            # number of times a feat is chosen as a reason
            if feature:
                num_a_feat_chosen[feature] += 1
            
            if selection_strategy == 'covering' or selection_strategy == 'covering_fewest':
                doc_pick_model.update(X_pool, feature, doc)
            elif selection_strategy == 'cheating':
                doc_pick_model.update(X_pool, feature, y_pool[doc])
            elif selection_strategy.startswith('cover_then') and doc_pick_model.phase == 'covering':
                doc_pick_model.covering.update(X_pool, feature, doc)
                    
        pooling_model.fit(instance_model, feature_model, weights=[0.5, 0.5]) # train pooling_model
        
        (accu, auc) = evaluate_model(instance_model, X_test, y_test)
        instance_model_scores['auc'].append(auc)
        instance_model_scores['accu'].append(accu)
        
        (accu, auc) = evaluate_model(feature_model, X_test, y_test)
        feature_model_scores['auc'].append(auc)
        feature_model_scores['accu'].append(accu)
        
        (accu, auc) = evaluate_model(pooling_model, X_test, y_test)
        pooling_model_scores['auc'].append(auc)
        pooling_model_scores['accu'].append(accu)
        
        (accu, auc) = evaluate_model(reasoning_model, X_test, y_test)
        reasoning_model_scores['auc'].append(auc)
        reasoning_model_scores['accu'].append(accu)
        
        # discovered feature counts
        if isinstance(feature_model, FeatureMNBUniform):        
            discovered_feature_counts['class0'].append(len(feature_model.class0_features))
            discovered_feature_counts['class1'].append(len(feature_model.class1_features))
        elif isinstance(feature_model, FeatureMNBWeighted):
            nz = np.sum(feature_model.feature_count_>0, axis=1)
            discovered_feature_counts['class0'].append(nz[0])
            discovered_feature_counts['class1'].append(nz[1])
        
        num_docs_covered.append(len(covered_docs))
    
    else:
        if selection_strategy.startswith('uncertainty') or selection_strategy == 'disagreement':
            raise ValueError('\'%s\' requires bootstrapping!' % selection_strategy)            
       

    for i in range(budget):
        train_set_size=len(training_set)

        # Choose a document based on the strategy chosen
        if selection_strategy.startswith('cover_then'):
            doc_id = doc_pick_model.choice(X_pool, i+1, pool_set)        
        elif selection_strategy.startswith('optauc'):
            doc_id = doc_pick_model.choice(X_pool, y_pool, pool_set, training_set, feature_model, reasoning_model, rmw_n, rmw_a)
        elif selection_strategy == 'reasoning_then_featureCertainty':
            doc_id = doc_pick_model.choice(X_pool, i+1, pool_set, train_set_size)
        elif selection_strategy == "unc_insuff_R":
            doc_id = doc_pick_model.choice(X_pool, pool_set, discovered_features, max_num_feats=1)
        elif selection_strategy == "unc_no_conflict_R":
            doc_id = doc_pick_model.choice(X_pool, pool_set, discovered_class0_features, discovered_class1_features)
        elif selection_strategy == "unc_prefer_no_conflict_R":
            doc_id = doc_pick_model.choice(X_pool, pool_set, discovered_class0_features, discovered_class1_features, top_k=10)
        elif selection_strategy == "unc_prefer_conflict_R":
            doc_id = doc_pick_model.choice(X_pool, pool_set, discovered_class0_features, discovered_class1_features, top_k=10)
        else:
            doc_id = doc_pick_model.choice(X_pool, pool_set)
        
        if doc_id == None:
            break
        
        # Remove the chosen document from pool and add it to the training set
        pool_set.remove(doc_id)
        training_set.append(doc_id)
        
        if i == 0 and training_set_empty:
            X_train = X_pool[doc_id]
            y_train = np.array([y_pool[doc_id]])
        else:
            X_train = sp.vstack((X_train, X_pool[doc_id]))
            y_train = np.hstack((y_train, np.array([y_pool[doc_id]])))
        
        # Ask the expert for instance label (returns the true label from the dataset)
        label = y_pool[doc_id]
        
        # Ask the expert for most informative feature given the label
        #feature = feature_expert.most_informative_feature(X_pool[doc_id], label)
        feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])
        
        # Update the instance model
        instance_model.fit(X_train, y_train)
        
        # Update the feature model
        if feature:
            feature_model.fit(feature, label)
            discovered_features.add(feature)
            if y_pool[doc_id] == 0:
                discovered_class0_features.add(feature)
            else:
                discovered_class1_features.add(feature)
            
        reasoning_model.partial_fit(X_pool[doc_id], y_pool[doc_id], feature, rmw_n, rmw_a) # train feature_model one by one
        
        # docs covered
        if feature:
            f_covered_docs = X_pool_csc[:, feature].indices
            covered_docs.update(f_covered_docs)
        
        # number of times a feat is chosen as a reason
        if feature:
            num_a_feat_chosen[feature] += 1
        
        
        # Update the pooling model
        pooling_model.fit(instance_model, feature_model, weights=[0.5, 0.5])
        
        # print 'docs = %d, feature = %s' % (doc_id, str(feature))
        
        if selection_strategy == 'covering' or selection_strategy == 'covering_fewest':
            doc_pick_model.update(X_pool, feature, doc_id)
        elif selection_strategy == 'cheating':
            doc_pick_model.update(X_pool, feature, label)
        elif selection_strategy.startswith('cover_then') and doc_pick_model.phase == 'covering':
            doc_pick_model.covering.update(X_pool, feature, doc_id)
                
#        print 'covering_fewest features: %d, feature model features: %d' % (len(doc_pick_model.annotated_features), len(feature_model.class0_features + feature_model.class1_features))

        # Evaluate performance based on Instance Model
        (accu, auc) = evaluate_model(instance_model, X_test, y_test)
        instance_model_scores['auc'].append(auc)
        instance_model_scores['accu'].append(accu)
        
        # Evaluate performance on Feature Model
        (accu, auc) = evaluate_model(feature_model, X_test, y_test)
        feature_model_scores['auc'].append(auc)
        feature_model_scores['accu'].append(accu)
        
        # Evaluate performance on Pooled Model
        (accu, auc) = evaluate_model(pooling_model, X_test, y_test)
        pooling_model_scores['auc'].append(auc)
        pooling_model_scores['accu'].append(accu)
        
        # Evaluate performance of the Reasoning Model
        (accu, auc) = evaluate_model(reasoning_model, X_test, y_test)
        reasoning_model_scores['auc'].append(auc)
        reasoning_model_scores['accu'].append(accu)
        
        # discovered feature counts
        if isinstance(feature_model, FeatureMNBUniform):        
            discovered_feature_counts['class0'].append(len(feature_model.class0_features))
            discovered_feature_counts['class1'].append(len(feature_model.class1_features))
        elif isinstance(feature_model, FeatureMNBWeighted):
            nz = np.sum(feature_model.feature_count_>0, axis=1)
            discovered_feature_counts['class0'].append(nz[0])
            discovered_feature_counts['class1'].append(nz[1])
         
        # docs covered        
        num_docs_covered.append(len(covered_docs))
    
    if selection_strategy.startswith('cover_then'):
        transition = doc_pick_model.transition
    else:
        transition = None
    
    
    # compute the # of training samples for plot
    if training_set_empty:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + 1
    else:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + bootstrap_size
    
    print 'Active Learning took %2.2fs' % (time() - start)
    
    return (num_training_samples, instance_model_scores, feature_model_scores, pooling_model_scores,reasoning_model_scores, discovered_feature_counts, num_docs_covered, transition, num_a_feat_chosen)

def load_dataset(dataset):
    if dataset == ['imdb']:
        #(X_pool, y_pool, X_test, y_test) = load_data()
        #vect = CountVectorizer(min_df=0.005, max_df=1./3, binary=True, ngram_range=(1,1))
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))        
        X_pool, y_pool, X_test, y_test, _, _, = load_imdb(path='C:\\Users\\mbilgic\\Desktop\\aclImdb', shuffle=True, vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif isinstance(dataset, list) and len(dataset) == 3 and dataset[0] == '20newsgroups':
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
        X_pool, y_pool, X_test, y_test, _, _ = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif dataset == ['SRAA']:
        X_pool = pickle.load(open('SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('SRAA_y_train.pickle', 'rb'))
        X_test = pickle.load(open('SRAA_X_test.pickle', 'rb'))
        y_test = pickle.load(open('SRAA_y_test.pickle', 'rb'))
        feat_names = pickle.load(open('SRAA_feature_names.pickle', 'rb'))
        return (X_pool, y_pool, X_test, y_test, feat_names)
    elif dataset == ['nova']:
        (X_pool, y_pool, X_test, y_test) = load_nova()
        return (X_pool, y_pool, X_test, y_test, None)
    
def run_trials(num_trials, dataset, selection_strategy, metric, C, alpha, smoothing, \
                bootstrap_size, balance, coverage, disagree_strat, budget, fmtype, rmw_n, rmw_a, seed=0, Debug=False, \
                reasoning_strategy='random', switch=40):
    
    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset(dataset)
    
    if not feat_names:
        feat_names = np.arange(X_pool.shape[1])
    
    feat_freq = np.diff(X_pool.tocsc().indptr)   
    
    fe = feature_expert(X_pool, y_pool, metric, smoothing=1e-6, C=C, pick_only_top=True)
    result = np.ndarray(num_trials, dtype=object)
    
    for i in range(num_trials):
        print '-' * 50
        print 'Starting Trial %d of %d...' % (i + 1, num_trials)

        trial_seed = seed + i # initialize the seed for the trial
        
        instance_model = MultinomialNB(alpha=alpha)
        
        feature_model = None 
        if fmtype == "fm_uniform":
            feature_model = FeatureMNBUniform([], [], fe.num_features, smoothing)
        elif fmtype == "fm_weighted":
            feature_model = FeatureMNBWeighted(num_feat = fe.num_features, imaginary_counts = 1.)
        else:
            raise ValueError('Feature model type: \'%s\' invalid!' % fmtype)
            
        pooling_model = PoolingMNB()
        
        reasoning_model = ReasoningMNB(alpha=1)

        if bootstrap_size == 0:
            training_set = []
            pool_set = range(X_pool.shape[0])
        else:
            training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
        
        result[i] = learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, \
            fe, selection_strategy, disagree_strat, coverage, budget, instance_model, \
            feature_model, pooling_model, reasoning_model, rmw_n, rmw_a, trial_seed, Debug, \
            reasoning_strategy, switch)
    
    return result, feat_names, feat_freq

def average_results(result):
    avg_IM_scores = dict()
    avg_FM_scores = dict()
    avg_PM_scores = dict()
    avg_RM_scores = dict()
    
    avg_discovered_feature_counts = dict()
    num_docs_covered = []
    
    num_trials = result.shape[0]
    
    if num_trials == 1:
        num_training_set, IM_scores, FM_scores, PM_scores, RM_scores, feature_counts, covered_docs, transition, num_a_feat_chosen = result[0]
        return np.array([(num_training_set, IM_scores, FM_scores, PM_scores, RM_scores, feature_counts, \
                          covered_docs, [transition], num_a_feat_chosen)])
    
    ls_transitions = []
    
    min_training_samples = np.inf
    for i in range(num_trials):
        # result[i][0] is the num_training_set
        min_training_samples = min(result[i][0].shape[0], min_training_samples)
    
    for i in range(num_trials):
        num_training_set, IM_scores, FM_scores, PM_scores, RM_scores, feature_counts, covered_docs, transition, num_a_feat_chosen = result[i]
        if i == 0:
            avg_IM_scores['accu'] = np.array(IM_scores['accu'])[:min_training_samples]
            avg_IM_scores['auc'] = np.array(IM_scores['auc'])[:min_training_samples]
            avg_FM_scores['accu'] = np.array(FM_scores['accu'])[:min_training_samples]
            avg_FM_scores['auc'] = np.array(FM_scores['auc'])[:min_training_samples]
            avg_PM_scores['accu'] = np.array(PM_scores['accu'])[:min_training_samples]
            avg_PM_scores['auc'] = np.array(PM_scores['auc'])[:min_training_samples]
            avg_RM_scores['accu'] = np.array(RM_scores['accu'])[:min_training_samples]
            avg_RM_scores['auc'] = np.array(RM_scores['auc'])[:min_training_samples]
            avg_discovered_feature_counts['class0'] = np.array(feature_counts['class0'])[:min_training_samples]
            avg_discovered_feature_counts['class1'] = np.array(feature_counts['class1'])[:min_training_samples]
            num_docs_covered = np.array(covered_docs)[:min_training_samples]
            ave_num_a_feat_chosen = np.array(num_a_feat_chosen)[:min_training_samples]
        else:
            avg_IM_scores['accu'] += np.array(IM_scores['accu'])[:min_training_samples]
            avg_IM_scores['auc'] += np.array(IM_scores['auc'])[:min_training_samples]
            avg_FM_scores['accu'] += np.array(FM_scores['accu'])[:min_training_samples]
            avg_FM_scores['auc'] += np.array(FM_scores['auc'])[:min_training_samples]
            avg_PM_scores['accu'] += np.array(PM_scores['accu'])[:min_training_samples]
            avg_PM_scores['auc'] += np.array(PM_scores['auc'])[:min_training_samples]
            avg_RM_scores['accu'] += np.array(RM_scores['accu'])[:min_training_samples]
            avg_RM_scores['auc'] += np.array(RM_scores['auc'])[:min_training_samples]
            avg_discovered_feature_counts['class0'] += np.array(feature_counts['class0'])[:min_training_samples]
            avg_discovered_feature_counts['class1'] += np.array(feature_counts['class1'])[:min_training_samples]
            num_docs_covered += np.array(covered_docs)[:min_training_samples]
            ave_num_a_feat_chosen += num_a_feat_chosen[:min_training_samples]
        
        ls_transitions.append(transition)
            
    num_training_set = num_training_set[:min_training_samples]
    avg_IM_scores['accu'] = avg_IM_scores['accu'] / num_trials
    avg_IM_scores['auc'] = avg_IM_scores['auc'] / num_trials
    avg_FM_scores['accu'] = avg_FM_scores['accu'] / num_trials
    avg_FM_scores['auc'] = avg_FM_scores['auc'] / num_trials
    avg_PM_scores['accu'] = avg_PM_scores['accu'] / num_trials
    avg_PM_scores['auc'] = avg_PM_scores['auc'] / num_trials
    avg_RM_scores['accu'] = avg_RM_scores['accu'] / num_trials
    avg_RM_scores['auc'] = avg_RM_scores['auc'] / num_trials
    avg_discovered_feature_counts['class0'] = avg_discovered_feature_counts['class0'] / float(num_trials)
    avg_discovered_feature_counts['class1'] = avg_discovered_feature_counts['class1'] / float(num_trials)
    num_docs_covered = num_docs_covered / float(num_trials)
    ave_num_a_feat_chosen = ave_num_a_feat_chosen / float(num_trials)
    
    if len(ls_transitions) != result.shape[0]:
        raise ValueError('Something is wrong with the transition numbers')
    if ls_transitions == [[] for i in range(result.shape[0])]:
        ls_transitions = []
    
    return np.array([(num_training_set, avg_IM_scores, avg_FM_scores, avg_PM_scores, avg_RM_scores, avg_discovered_feature_counts, num_docs_covered, ls_transitions, ave_num_a_feat_chosen)])

def plot(num_training_set, instance_model_scores, feature_model_scores, pooling_model_scores):
    axes_params = [0.1, 0.1, 0.58, 0.75]
    bbox_anchor_coord=(1.02, 1)
    
    # Plot the results
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)
    ax.plot(num_training_set, instance_model_scores['accu'], label='Instance Model', color='r', ls=':')
    ax.plot(num_training_set, feature_model_scores['accu'], label='Feature Model', color='b', ls='-.')
    ax.plot(num_training_set, pooling_model_scores['accu'], label='Pooling Model', color='g', ls='--')
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel('# of training samples')
    plt.ylabel('Accuracy')
    plt.title('Active Learning with Reasoning')
    plt.show()
    
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)
    ax.plot(num_training_set, instance_model_scores['auc'], label='Instance Model', color='r', ls=':')
    ax.plot(num_training_set, feature_model_scores['auc'], label='Feature Model', color='b', ls='-.')
    ax.plot(num_training_set, pooling_model_scores['auc'], label='Pooling Model', color='g', ls='--')
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel('# of training samples')
    plt.ylabel('AUC')
    plt.title('Active Learning with Reasoning')
    plt.show()

def full_knowledge(dataset, metric='mutual_info', C=0.1, alpha=1, smoothing=0):
    '''
    This function uses the entire pool to learn the instance model,
    feature model, and pooling model which provides an upper bound on how well
    these models can perform on this particular dataset
    '''
    (X_pool,  y_pool, X_test, y_test, feat_names) = load_dataset(dataset)
    fe = feature_expert(X_pool, y_pool, metric, smoothing=1e-6, C=C)
    
    instance_model_scores = {'auc':[], 'accu':[]}
    feature_model_scores = {'auc':[], 'accu':[]}
    pooling_model_scores = {'auc':[], 'accu':[]}
    
    instance_model = MultinomialNB(alpha=alpha)
    feature_model = FeatureMNBUniform([], [], fe.num_features, smoothing)
    pooling_model = PoolingMNB()

    instance_model.fit(X_pool, y_pool)
    for doc in range(X_pool.shape[0]):
        feature = fe.most_informative_feature(X_pool[doc], y_pool[doc])
        feature_model.fit(feature, y_pool[doc])

    # Evaluate performance based on Instance Model
    (accu, auc) = evaluate_model(instance_model, X_test, y_test)
    instance_model_scores['auc'].append(auc)
    instance_model_scores['accu'].append(accu)
    print 'Instance Model: auc = %f, accu = %f' % (auc, accu)
    
    # Evaluate performance on Feature Model
    (accu, auc) = evaluate_model(feature_model, X_test, y_test)
    feature_model_scores['auc'].append(auc)
    feature_model_scores['accu'].append(accu)
    print 'Feature Model: auc = %f, accu = %f' % (auc, accu)
    
    # Evaluate performance on Pooled Model
    pooling_model.fit(instance_model, feature_model, weights=[0.5, 0.5])
    (accu, auc) = evaluate_model(pooling_model, X_test, y_test)
    pooling_model_scores['auc'].append(auc)
    pooling_model_scores['accu'].append(accu)
    print 'Pooled Model: auc = %f, accu = %f' % (auc, accu)

def save_result(result, filename='result.txt'):
    # Saves the data the following order:
    # training sample index, IM_accu, FM_accu, PM_accu, IM_acu, FM_auc, PM_auc, c0_features_discovered so far,
    # c1_features_discovered so far, num_docs_covered, and transition phases for cover_then_disagree approach
    # if the approach is not cover_then_disagree, no transition is saved
    print '-' * 50
    print 'saving result into \'%s\'' % filename
    
    ls_all_results = []
    ls_transitions = []
    with open(filename, 'w') as f:
        for i in range(result.shape[0]):
            num_training_set, instance_model_scores, feature_model_scores, pooling_model_scores, \
            reasoning_model_scores, feature_counts, covered_docs, transition, num_a_feat_chosen = result[i]

            ls_all_results.append(num_training_set)
            ls_all_results.append(instance_model_scores['accu'])
            ls_all_results.append(feature_model_scores['accu'])
            ls_all_results.append(pooling_model_scores['accu'])
            ls_all_results.append(reasoning_model_scores['accu'])
            ls_all_results.append(instance_model_scores['auc'])
            ls_all_results.append(feature_model_scores['auc'])
            ls_all_results.append(pooling_model_scores['auc'])
            ls_all_results.append(reasoning_model_scores['auc'])
            ls_all_results.append(feature_counts['class0'])
            ls_all_results.append(feature_counts['class1'])
            ls_all_results.append(covered_docs)
            if result.shape[0] == 1 and isinstance(transition, list):
                ls_all_results.append(ls_transitions)
            else:
                ls_all_results.append([transition])
        
        header = 'train#\tIM_accu\tFM_accu\tPM_accu\tRM_accu\tIM_auc\tFM_auc\tPM_auc\tRM_auc\tc0_feat\tc1_feat\tdocs_covered\ttransition'
        f.write('\t'.join([header]*result.shape[0]) + '\n')
        for row in map(None, *ls_all_results):
            f.write('\t'.join([str(item) if item is not None else ' ' for item in row]) + '\n')

def nparray_tostr(array):
    return ' '.join([str(val) for val in array]) + '\n'
        
def load_result(filename='result.txt'):
    # This file currently loads only the accuracies and aucs. The discovered feature counts
    # and number of covered documents is not loaded.
    instance_model_scores = dict()
    feature_model_scores = dict()
    pooling_model_scores = dict()

    print '-' * 50
    print 'loading result from \'%s\'' % filename
    with open(filename, 'r') as f:
        num_training_set = np.array(f.readline().strip('\n').split(), dtype='float')
        instance_model_scores['accu'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        feature_model_scores['accu'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        pooling_model_scores['accu'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        instance_model_scores['auc'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        feature_model_scores['auc'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        pooling_model_scores['auc'] = np.array([float(val) for val in f.readline().strip('\n').split()])
    
    return (num_training_set, instance_model_scores, feature_model_scores, pooling_model_scores)

def evaluate_model(model, X_test, y_test):
    # model must have predict_proba, classes_
    y_probas = model.predict_proba(X_test)
    auc = metrics.roc_auc_score(y_test, y_probas[:, 1])
    pred_y = model.classes_[np.argmax(y_probas, axis=1)]
    accu = metrics.accuracy_score(y_test, pred_y)
    return (accu, auc)

def save_result_num_a_feat_chosen(result, feat_names, feat_freq):
    
    ave_num_a_feat_chosen = np.zeros(len(feat_names))
    
    for i in range(args.trials):
        num_a_feat_chosen = result[i][-1]
        ave_num_a_feat_chosen += (num_a_feat_chosen / float(args.trials))
    
    filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:0.2f}coverage'.format(args.coverage), 'w_a={:0.2f}'.format(args.rmw_a), 'w_n={:0.2f}'.format(args.rmw_n), "num_a_feat_chosen", 'result.txt'])
    
    print '-' * 50
    print 'saving result into \'%s\'' % filename
            
    with open(filename, 'w') as f:
        f.write("ID\tNAME\tFREQ\tCOUNT\n")
        for i in range(len(feat_names)):
            f.write(str(i)+"\t"+feat_names[i].encode('utf8')+"\t"+str(feat_freq[i])+"\t"+str(ave_num_a_feat_chosen[i])+"\n")

if __name__ == '__main__':
    '''
    Example: 
    To run covering approach with L1 feature expert for 10 trials, no bootstrap and a budget of $100:
    python active_learn.py -strategy covering_fewest -metric L1 -trials 10 -bootstrap 0 -budget 100
    python active_learn.py -dataset 20newsgroups comp.graphics comp.windows.x -strategy random -metric L1 -trials 10 -bootstrap 0 -budget 500
    python active_learn.py -dataset 20newsgroups alt.atheism talk.religion.misc -strategy random -metric mutual_info -trials 10 -bootstrap 0 -budget 500
    python active_learn.py -dataset 20newsgroups alt.atheism talk.religion.misc -strategy cover_then_disagree -metric L1 -trials 10 -bootstrap 0 -budget 500
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default=['imdb'], nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-strategy', choices=['random', 'uncertaintyIM', 'uncertaintyFM', \
                        'uncertaintyPM', 'uncertaintyRM', 'disagreement', 'covering', 'covering_fewest', \
                        'cheating', 'cover_then_disagree', 'cover_then_uncertainty', \
                        'optaucP', 'optaucI', 'optaucF', 'optaucR', 'reasoning_then_featureCertainty', \
                        'cover_then_uncertaintyRM', 'cover_then_featureCertainty', 'unc_insuff_R', \
                        'unc_no_conflict_R', 'unc_prefer_no_conflict_R', 'unc_prefer_conflict_R'], default='random', \
                        help='Document selection strategy to be used')
    parser.add_argument('-reasoningStrategy', choices=['random', 'uncertaintyIM', 'uncertaintyPM'], default='random', \
                        help='Reasoning strategy to be used for reasoning_then_disagreement')
    parser.add_argument('-switch', type=int, default=38, help='After how many documents to switch from reasoning to FM Uncertainty')
    parser.add_argument('-metric', choices=['mutual_info', 'chi2', 'L1'], default="L1", \
                        help='Specifying the type of feature expert to be used')
    parser.add_argument('-c', type=float, default=0.1, help='Penalty term for the L1 feature expert')
    parser.add_argument('-debug', action='store_true', help='Enable Debugging')
    parser.add_argument('-trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('-seed', type=int, default=0, help='Seed to the random number generator')
    parser.add_argument('-bootstrap', type=int, default=2, help='Number of documents to bootstrap')
    parser.add_argument('-balance', default=True, action='store_false', help='Ensure both classes starts with equal # of docs after bootstrapping')
    parser.add_argument('-budget', type=int, default=500, help='budget in $')
    parser.add_argument('-alpha', type=float, default=1, help='alpha for the MultinomialNB instance model')
    parser.add_argument('-cost', type=float, default=1, help='cost per document for (class label + feature label)')
    parser.add_argument('-smoothing', type=float, default=0, help='smoothing parameter for the feature MNB model')
    parser.add_argument('-coverage', type=float, default=1., help='% docs covered before disagreement is ran')
    parser.add_argument('-disagree_metric', default='KLD', help='metric used to determine disagreement between IM and FM')
    parser.add_argument('-fmtype', choices=['fm_uniform', 'fm_weighted'], default="fm_weighted", help='The feature model type to use')
    parser.add_argument('-rmw_n', type=float, default=1., help='The weight of non-annotated features for the reasoning model')
    parser.add_argument('-rmw_a', type=float, default=1., help='The weight of annotated features for the reasoning model')

    args = parser.parse_args()

    result, feat_names, feat_freq = run_trials(num_trials=args.trials, dataset=args.dataset, selection_strategy=args.strategy,\
                metric=args.metric, C=args.c, alpha=args.alpha, smoothing=args.smoothing, \
                bootstrap_size=args.bootstrap, balance=args.balance, coverage=args.coverage, \
                disagree_strat=args.disagree_metric, budget=args.budget/args.cost, \
                fmtype=args.fmtype, rmw_n=args.rmw_n, rmw_a=args.rmw_a, seed=args.seed, Debug=args.debug, \
                reasoning_strategy=args.reasoningStrategy, switch=args.switch)
    
    if args.strategy.startswith('cover_then_'):
        save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:0.2f}coverage'.format(args.coverage), '{:d}trials'.format(args.trials), 'w_a={:0.2f}'.format(args.rmw_a), 'w_n={:0.2f}'.format(args.rmw_n), 'result.txt']))
        save_result(average_results(result), filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:0.2f}coverage'.format(args.coverage), 'w_a={:0.2f}'.format(args.rmw_a), 'w_n={:0.2f}'.format(args.rmw_n), 'averaged', 'result.txt']))    
    else:
        save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:d}trials'.format(args.trials), 'w_a={:0.2f}'.format(args.rmw_a), 'w_n={:0.2f}'.format(args.rmw_n), 'result.txt']))
        save_result(average_results(result), filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, 'w_a={:0.2f}'.format(args.rmw_a), 'w_n={:0.2f}'.format(args.rmw_n), 'averaged', 'result.txt']))
    
    save_result_num_a_feat_chosen(result, feat_names, feat_freq)

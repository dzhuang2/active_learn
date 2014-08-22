from time import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from imdb import load_imdb, load_newsgroups, load_nova
from models import FeatureMNB, PoolingMNB
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from feature_expert import feature_expert
from selection_strategies import RandomBootstrap, RandomStrategy, UNCSampling, DisagreementStrategy
from selection_strategies import CoveringStrategy, CheatingApproach, CoveringThenDisagreement

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
          selection_strategy, coverage, budget, instance_model, feature_model, \
          pooling_model, seed=0, Debug=False):
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    instance_model_scores = {'auc':[], 'accu':[]}
    feature_model_scores = {'auc':[], 'accu':[]}
    pooling_model_scores = {'auc':[], 'accu':[]}
        
    discovered_feature_counts = {'class0':[], 'class1': []}
    num_docs_covered = []    
    covered_docs = set()    
    X_pool_csc = X_pool.tocsc()
    
    num_samples = len(pool_set) + len(training_set)
    
    if selection_strategy == 'random':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'uncertaintyIM':
        doc_pick_model = UNCSampling(instance_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'uncertaintyFM':
        doc_pick_model = UNCSampling(feature_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'uncertaintyPM':
        doc_pick_model = UNCSampling(pooling_model, feature_expert, y_pool, Debug)
    elif selection_strategy == 'disagreement':
        doc_pick_model = DisagreementStrategy(instance_model, feature_model, \
            feature_expert, y_pool, 'KLD', Debug=Debug)
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
            metric='KLD', seed=seed, Debug=Debug)
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
            feature = feature_expert.most_informative_feature(X_pool[doc], y_pool[doc])
            
            if feature:
                feature_model.fit(feature, y_pool[doc]) # train feature_model one by one
            
            # docs covered            
            if feature:
                f_covered_docs = X_pool_csc[:, feature].indices
                covered_docs.update(f_covered_docs)
            
            if selection_strategy == 'covering' or selection_strategy == 'covering_fewest':
                doc_pick_model.update(X_pool, feature)
            elif selection_strategy == 'cheating':
                doc_pick_model.update(X_pool, feature, y_pool[doc])
            elif selection_strategy == 'cover_then_disagree' and doc_pick_model.phase == 'covering':
                doc_pick_model.covering.update(X_pool, feature)
        
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
        
        # discovered feature counts
        discovered_feature_counts['class0'].append(len(feature_model.class0_features))
        discovered_feature_counts['class1'].append(len(feature_model.class1_features))
        
        num_docs_covered.append(len(covered_docs))
    
    else:
        if selection_strategy.startswith('uncertainty') or selection_strategy == 'disagreement':
            raise ValueError('\'%s\' requires bootstrapping!' % selection_strategy)            
    
    for i in range(budget):
        # Choose a document based on the strategy chosen
        if selection_strategy == 'cover_then_disagree':
            doc_id = doc_pick_model.choice(X_pool, i+1, pool_set)
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
        feature = feature_expert.most_informative_feature(X_pool[doc_id], label)
        
        # Update the instance model
        instance_model.fit(X_train, y_train)
        
        # Update the feature model
        if feature:
            feature_model.fit(feature, label)
        
        # docs covered
        if feature:
            f_covered_docs = X_pool_csc[:, feature].indices
            covered_docs.update(f_covered_docs)
        
        # Update the pooling model
        pooling_model.fit(instance_model, feature_model, weights=[0.5, 0.5])
        
        if selection_strategy == 'covering' or selection_strategy == 'covering_fewest':
            doc_pick_model.update(X_pool, feature)
        elif selection_strategy == 'cheating':
            doc_pick_model.update(X_pool, feature, label)
        elif selection_strategy == 'cover_then_disagree' and doc_pick_model.phase == 'covering':
            doc_pick_model.covering.update(X_pool, feature)
        
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
        
        # discovered feature counts
        discovered_feature_counts['class0'].append(len(feature_model.class0_features))
        discovered_feature_counts['class1'].append(len(feature_model.class1_features))
        
        # docs covered        
        num_docs_covered.append(len(covered_docs))
    
    if isinstance(doc_pick_model, CoveringThenDisagreement):
        transition = doc_pick_model.transition
        print 'covering transition happens at sample #%d' % doc_pick_model.transition
    else:
        transition = None
    
    # compute the # of training samples for plot
    if training_set_empty:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + 1
    else:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + bootstrap_size
    
    print 'Active Learning took %2.2fs' % (time() - start)
    
    return (num_training_samples, instance_model_scores, feature_model_scores, pooling_model_scores, discovered_feature_counts, num_docs_covered, transition)

def load_dataset(dataset):
    if dataset == ['imdb']:
        (X_pool, y_pool, X_test, y_test) = load_data()
        return (X_pool, y_pool, X_test, y_test)
    elif isinstance(dataset, list) and len(dataset) == 3 and dataset[0] == '20newsgroups':
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 3))
        X_pool, y_pool, X_test, y_test, X_pool_docs, X_test_docs = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test)
    elif dataset == ['SRAA']:
        X_pool = pickle.load(open('SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('SRAA_y_train.pickle', 'rb'))
        X_test = pickle.load(open('SRAA_X_test.pickle', 'rb'))
        y_test = pickle.load(open('SRAA_y_test.pickle', 'rb'))
        return (X_pool, y_pool, X_test, y_test)
    elif dataset == ['nova']:
        (X_pool, y_pool, X_test, y_test) = load_nova()
        return (X_pool, y_pool, X_test, y_test)
    
def run_trials(num_trials, dataset, selection_strategy, metric, C, alpha, smoothing, \
                bootstrap_size, balance, coverage, budget, seed=0, Debug=False):
    
    (X_pool, y_pool, X_test, y_test) = load_dataset(dataset)    
    fe = feature_expert(X_pool, y_pool, metric, smoothing=1e-6, C=C)
    result = np.ndarray(num_trials, dtype=object)
    
    for i in range(num_trials):
        print '-' * 50
        print 'Starting Trial %d of %d...' % (i + 1, num_trials)

        trial_seed = seed + i # initialize the seed for the trial
        
        instance_model = MultinomialNB(alpha=alpha)
        feature_model = FeatureMNB([], [], fe.num_features, smoothing)
        pooling_model = PoolingMNB()

        if bootstrap_size == 0:
            training_set = []
            pool_set = range(X_pool.shape[0])
        else:
            training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
        
        result[i] = learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, \
            fe, selection_strategy, coverage, budget, instance_model, feature_model, pooling_model, trial_seed, Debug)
    
    return result

def average_results(result):
    avg_IM_scores = dict()
    avg_FM_scores = dict()
    avg_PM_scores = dict()
    
    avg_discovered_feature_counts = dict()
    num_docs_covered = []
    
    num_trials = result.shape[0]
    
    if num_trials == 1:
        return result[0]
    
    ls_transitions = []
    
    for i in range(num_trials):
        num_training_set, IM_scores, FM_scores, PM_scores, feature_counts, covered_docs, transition = result[i]
        if i == 0:
            avg_IM_scores['accu'] = np.array(IM_scores['accu'])
            avg_IM_scores['auc'] = np.array(IM_scores['auc'])
            avg_FM_scores['accu'] = np.array(FM_scores['accu'])
            avg_FM_scores['auc'] = np.array(FM_scores['auc'])
            avg_PM_scores['accu'] = np.array(PM_scores['accu'])
            avg_PM_scores['auc'] = np.array(PM_scores['auc'])
            avg_discovered_feature_counts['class0'] = np.array(feature_counts['class0'])
            avg_discovered_feature_counts['class1'] = np.array(feature_counts['class1'])
            num_docs_covered = np.array(covered_docs)
        else:
            avg_IM_scores['accu'] += np.array(IM_scores['accu'])
            avg_IM_scores['auc'] += np.array(IM_scores['auc'])
            avg_FM_scores['accu'] += np.array(FM_scores['accu'])
            avg_FM_scores['auc'] += np.array(FM_scores['auc'])
            avg_PM_scores['accu'] += np.array(PM_scores['accu'])
            avg_PM_scores['auc'] += np.array(PM_scores['auc'])
            avg_discovered_feature_counts['class0'] += np.array(feature_counts['class0'])
            avg_discovered_feature_counts['class1'] += np.array(feature_counts['class1'])
            num_docs_covered += np.array(covered_docs)
        
        ls_transitions.append(transition)
            
    avg_IM_scores['accu'] = avg_IM_scores['accu'] / num_trials
    avg_IM_scores['auc'] = avg_IM_scores['auc'] / num_trials
    avg_FM_scores['accu'] = avg_FM_scores['accu'] / num_trials
    avg_FM_scores['auc'] = avg_FM_scores['auc'] / num_trials
    avg_PM_scores['accu'] = avg_PM_scores['accu'] / num_trials
    avg_PM_scores['auc'] = avg_PM_scores['auc'] / num_trials
    avg_discovered_feature_counts['class0'] = avg_discovered_feature_counts['class0'] / float(num_trials)
    avg_discovered_feature_counts['class1'] = avg_discovered_feature_counts['class1'] / float(num_trials)
    num_docs_covered = num_docs_covered / float(num_trials)
    
    if len(ls_transitions) != result.shape[0]:
        raise ValueError('Something is wrong with the transition numbers')
    if ls_transitions == [[] for i in range(result.shape[0])]:
        ls_transitions = []
    
    return np.array([(num_training_set, avg_IM_scores, avg_FM_scores, avg_PM_scores, avg_discovered_feature_counts, num_docs_covered, ls_transitions)])

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
    (X_pool,  y_pool, X_test, y_test) = load_dataset(dataset)
    fe = feature_expert(X_pool, y_pool, metric, smoothing=1e-6, C=C)
    
    instance_model_scores = {'auc':[], 'accu':[]}
    feature_model_scores = {'auc':[], 'accu':[]}
    pooling_model_scores = {'auc':[], 'accu':[]}
    
    instance_model = MultinomialNB(alpha=alpha)
    feature_model = FeatureMNB([], [], fe.num_features, smoothing)
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
    
    ls_transitions = []
    with open(filename, 'w') as f:
        for i in range(result.shape[0]):
            num_training_set, instance_model_scores, feature_model_scores, pooling_model_scores, feature_counts, covered_docs, transition = result[i]
            
            f.write(nparray_tostr(num_training_set))
            
            f.write(nparray_tostr(instance_model_scores['accu']))
            f.write(nparray_tostr(feature_model_scores['accu']))
            f.write(nparray_tostr(pooling_model_scores['accu']))
            
            f.write(nparray_tostr(instance_model_scores['auc']))
            f.write(nparray_tostr(feature_model_scores['auc']))
            f.write(nparray_tostr(pooling_model_scores['auc']))
            
            f.write(nparray_tostr(feature_counts['class0']))
            f.write(nparray_tostr(feature_counts['class1']))
            
            f.write(nparray_tostr(covered_docs))
            ls_transitions.append(transition)
        
        if ls_transitions == [[] for i in range(result.shape[0])]:
            f.write('\n')
        elif isinstance(ls_transitions[0], list): # meaning that the result is averaged
            f.write(nparray_tostr(np.array(ls_transitions[0])))
        else:
            f.write(nparray_tostr(np.array(ls_transitions)))

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
    parser.add_argument('-dataset', default='imdb', nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-strategy', choices=['random', 'uncertaintyIM', 'uncertaintyFM', \
                        'uncertaintyPM', 'disagreement', 'covering', 'covering_fewest', \
                        'cheating', 'cover_then_disagree'], default='random', \
                        help='Document selection strategy to be used')
    parser.add_argument('-metric', choices=['mutual_info', 'L1'], default="L1", \
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
    args = parser.parse_args()

    result = run_trials(num_trials=args.trials, dataset=args.dataset, selection_strategy=args.strategy,\
                metric=args.metric, C=args.c, alpha=args.alpha, smoothing=args.smoothing, \
                bootstrap_size=args.bootstrap, balance=args.balance, coverage=args.coverage, budget=args.budget/args.cost, \
                seed=args.seed, Debug=args.debug)
    
    if args.strategy == 'cover_then_disagree':
        save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:0.2f}coverage'.format(args.coverage), '{:d}trials'.format(args.trials), 'result.txt']))
        save_result(average_results(result), filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:0.2f}coverage'.format(args.coverage), 'averaged', 'result.txt']))
    else:
        save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:d}trials'.format(args.trials), 'result.txt']))
        save_result(average_results(result), filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, 'averaged', 'result.txt']))
    
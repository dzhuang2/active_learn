from time import time
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from active_learn import load_data, evaluate_model, nparray_tostr
from selection_strategies import RandomBootstrap, RandomStrategy, UNCSampling

def plot(num_training_set, instance_model_scores):
    axes_params = [0.1, 0.1, 0.58, 0.75]
    bbox_anchor_coord=(1.02, 1)
    
    # Plot the results
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)
    ax.plot(num_training_set, instance_model_scores['accu'], label='Instance Model', color='r', ls=':')
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel('# of training samples')
    plt.ylabel('Accuracy')
    plt.title('Active Learning with Reasoning')
    plt.show()
    
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)
    ax.plot(num_training_set, instance_model_scores['auc'], label='Instance Model', color='r', ls=':')
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel('# of training samples')
    plt.ylabel('AUC')
    plt.title('Active Learning with Reasoning')
    plt.show()

def average_results(result):
    avg_IM_scores = dict()
    num_trials = result.shape[0]
    
    if num_trials == 1:
        return result[0]
    
    for i in range(num_trials):
        num_training_set, IM_scores = result[i]
        if i == 0:
            avg_IM_scores['accu'] = np.array(IM_scores['accu'])
            avg_IM_scores['auc'] = np.array(IM_scores['auc'])
        else:
            avg_IM_scores['accu'] += np.array(IM_scores['accu'])
            avg_IM_scores['auc'] += np.array(IM_scores['auc'])
        
    avg_IM_scores['accu'] = avg_IM_scores['accu'] / num_trials
    avg_IM_scores['auc'] = avg_IM_scores['auc'] / num_trials
    
    return (num_training_set, avg_IM_scores)

def save_result(result, filename='result.txt'):
    print '-' * 50
    print 'saving result into \'%s\'' % filename
    with open(filename, 'w') as f:
        num_training_set, instance_model_scores = result
        f.write(nparray_tostr(num_training_set))
        f.write(nparray_tostr(instance_model_scores['accu']))
        f.write(nparray_tostr(instance_model_scores['auc']))

def load_result(filename='result.txt'):
    instance_model_scores = dict()

    print '-' * 50
    print 'loading result from \'%s\'' % filename
    with open(filename, 'r') as f:
        num_training_set = np.array(f.readline().strip('\n').split())
        instance_model_scores['accu'] = np.array([float(val) for val in f.readline().strip('\n').split()])
        instance_model_scores['auc'] = np.array([float(val) for val in f.readline().strip('\n').split()])
    
    return (num_training_set, instance_model_scores)

def no_reasoning_learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, \
        selection_strategy, budget, instance_model, seed=0, Debug=False):
    
    instance_model_scores = {'auc':[], 'accu':[]}
    
    if selection_strategy == 'random':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'uncertainty':
        doc_pick_model = UNCSampling(instance_model, None, y_pool, Debug)
    else:
        raise ValueError('Selection strategy: \'%s\' invalid!' % selection_strategy)
    
    bootstrap_size = len(training_set)
    training_set_empty = (bootstrap_size == 0)
    
    if not training_set_empty:
        X_train = X_pool[training_set]
        y_train = y_pool[training_set]
        instance_model.fit(X_train, y_train) # train instance_model
        (accu, auc) = evaluate_model(instance_model, X_test, y_test)
        instance_model_scores['auc'].append(auc)
        instance_model_scores['accu'].append(accu)
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    for i in range(budget):
        if Debug:
            print 'Sample %d of %d:' % (i+1, budget)
        
        # Choose a document based on the strategy chosen
        doc_id = doc_pick_model.choice(X_pool, pool_set)
        
        # Remove the chosen document from pool and add it to the training set
        pool_set.remove(doc_id)
        training_set.append(doc_id)
        
        if i == 0 and training_set_empty:
            X_train = X_pool[doc_id]
            y_train = np.array([y_pool[doc_id]])
        else:
            X_train=sp.vstack((X_train, X_pool[doc_id]))
            y_train=np.hstack((y_train, np.array([y_pool[doc_id]])))
        
        # Update the instance model
        instance_model.fit(X_train, y_train)
        
        # Evaluate model
        (accu, auc) = evaluate_model(instance_model, X_test, y_test)
        instance_model_scores['auc'].append(auc)
        instance_model_scores['accu'].append(accu)
        
        if Debug:
            print 'Instance Model: auc = %f, accu = %f' % (auc, accu)
        
    print 'Active Learning took %2.2fs' % (time() - start)
    
    # compute the # of training samples for plot
    if not training_set_empty:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + bootstrap_size
    else:
        num_training_samples = np.arange(len(instance_model_scores['accu'])) + 1
    
    return (num_training_samples, instance_model_scores)

def no_reasoning_run(num_trials, selection_strategy, alpha, bootstrap_size, \
                     balance, budget, seed=0, Debug=False):
    (X_pool, y_pool, X_test, y_test) = load_data()
    result = np.ndarray(num_trials, dtype=object)
    
    for i in range(num_trials):
        print '-' * 50
        print 'Starting Trial %d of %d...' % (i + 1, num_trials)

        trial_seed = seed + i
        instance_model = MultinomialNB(alpha=alpha)

        training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
        
        result[i] = no_reasoning_learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, \
            selection_strategy, budget, instance_model, trial_seed, Debug)
    
    return result

if __name__ == '__main__':
    '''
    strategies: 'random', 'uncertainty'
    '''
    # selection_strategy='random'
    selection_strategy='uncertainty'
    
    results = no_reasoning_run(num_trials=10, selection_strategy=selection_strategy, \
            alpha=1, bootstrap_size=2, balance=True, budget=1000, seed=0, Debug=False)
    
    save_result(average_results(results), filename='_'.join([selection_strategy, \
        'no_reasoning_result.txt']))
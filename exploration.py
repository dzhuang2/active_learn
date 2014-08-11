import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from imdb import load_imdb, load_newsgroups, load_sraa
from sklearn.linear_model import LogisticRegression
from active_learn import evaluate_model, load_dataset
from no_reasoning import average_results, save_result, no_reasoning_learn

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')

def IM_explore(num_trials, dataset, bootstrap_size=0, balance=True, budget=500, seed=2343, Debug=False):
    sep = '-' * 50
    
    (X_pool, y_pool, X_test, y_test) = load_dataset(dataset)
    
    models = {'MultinomialNB(alpha=1)':MultinomialNB(alpha=1), \
              'LogisticRegression(C=0.1, penalty='l1')':LogisticRegression(C=0.1, penalty='l1')), \
              'LogisticRegression(C=1, penalty='l1')':LogisticRegression(C=1, penalty='l1'))}
    
    print sep
    print 'Instance Model Performance Evaluation'
    
    model_result = np.ndarray(len(models), dtype=object)
    result = np.ndarray(num_trials, dtype=object)
    
    for model in models.keys():
        print sep
        print 'Instance Model: %s' % models[model]
        
        for i in range(num_trials):
            print sep
            print 'Starting Trial %d of %d...' % (i + 1, num_trials)

            trial_seed = seed + i # initialize the seed for the trial
            
            training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
            
            result[i] = no_reasoning_learn(X_pool, y_pool, X_test, y_test, training_set, pool_set, \
                'random', budget, models[model], trial_seed, Debug)
        
        model_result[model] = average_result(result)
        save_result(model_result[model], filename='_'.join(dataset, model, 'result.txt'))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='imdb', nargs='*', \
                        help='Dataset to be used: [\'imdb\', \'SRAA\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('-seed', type=int, default=2345, help='Seed to the random number generator')
    parser.add_argument('-bootstrap', type=int, default=2, help='Number of documents to bootstrap')
    parser.add_argument('-balance', action='store_true', help='Ensure both classes starts with equal # ' + \
                        'of docs after bootstrapping')
    parser.add_argument('-budget', type=int, default=250, help='budget in $')
    args = parser.parse_args()
    
    IM_explore(num_trials, dataset=args.dataset, bootstrap_size=args.bootstrap, balance=args.balance, budget=args.budget, seed=args.seed)
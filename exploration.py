import argparse
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from active_learn import load_dataset
from no_reasoning import average_results, save_result, no_reasoning_learn
from selection_strategies import RandomBootstrap

def IM_explore(num_trials, dataset, bootstrap_size=0, balance=True, budget=500, seed=2343, Debug=False):
    sep = '-' * 50
    
    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset(dataset)
    
    models = {'MultinomialNB(alpha=1)':MultinomialNB(alpha=1), \
              'LogisticRegression(C=0.1, penalty=\'l1\')':LogisticRegression(C=0.1, penalty='l1'), \
              'LogisticRegression(C=1, penalty=\'l1\')':LogisticRegression(C=1, penalty='l1')}
              
    # models = {'LogisticRegression(C=0.1, penalty=\'l1\')':LogisticRegression(C=0.1, penalty='l1')}

    print sep
    print 'Instance Model Performance Evaluation'
    
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
                'random', budget, models[model], trial_seed, Debug=Debug)
            
            # save_result(result[i], filename='_'.join([dataset, 'trial'+str(i), 'result.txt']))
        
        if isinstance(dataset, list):
            name = '_'.join(dataset)
            save_result(average_results(result), filename='_'.join([name, model, 'result.txt']))
        else:
            save_result(average_results(result), filename='_'.join([dataset, model, 'result.txt']))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', default='imdb', \
                        help='Dataset to be used: [\'imdb\', \'SRAA\', \'20newsgroups\'] 20newsgroups must have 2 valid group names')
    parser.add_argument('-cat', default=['alt.atheism', 'talk.religion.misc'], nargs=2, \
                        help='2 class labels from the 20newsgroup dataset')
    parser.add_argument('-trials', type=int, default=10, help='Number of trials to run')
    parser.add_argument('-seed', type=int, default=28, help='Seed to the random number generator')
    parser.add_argument('-bootstrap', type=int, default=2, help='Number of documents to bootstrap')
    parser.add_argument('-balance', default=True, action='store_false', help='Ensure both classes starts with equal # of docs after bootstrapping')
    parser.add_argument('-budget', type=int, default=500, help='budget in $')
    args = parser.parse_args()
    
    if args.dataset == '20newsgroups':
        dataset = [args.dataset, args.cat[0], args.cat[1]]
    else:
        dataset = args.dataset
    
    IM_explore(num_trials=args.trials, dataset=dataset, bootstrap_size=args.bootstrap, budget=args.budget, seed=args.seed)
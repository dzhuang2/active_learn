from time import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from imdb import load_imdb, load_newsgroups, load_nova, load_ibnsina, load_creditg
#from models import ReasoningMNB
from sklearn import metrics
from feature_expert import feature_expert
from selection_strategies_batch import RandomBootstrap, RandomStrategy, UNCSampling
from selection_strategies_batch import UNCPreferNoConflict, UNCPreferConflict, UNCThreeTypes
from scipy import stats
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from adaptive_lr import LogisticRegressionAdaptive
from sklearn.linear_model import SGDClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')
  

def learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, feature_expert, \
          selection_strategy, budget, step_size, topk, w_o, w_r, seed=0, lr_C=1, svm_C=1, svm_gamma=0, Debug=False):
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    
    model_scores = {'auc':[], 'accu':[]}
    
    rationales  = set()
    rationales_c0  = set()
    rationales_c1  = set()

    feature_expert.rg.seed(seed)
    
    num_training_samples = []
    
    
    docs = training_set
    
    X_train = None
    y_train = []
    
    for doc_id in docs:
        #feature = feature_expert.most_informative_feature(X_pool[doc_id], y_pool[doc_id])
        feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])
        
        rationales.add(feature)

        if y_pool[doc_id] == 0:
            rationales_c0.add(feature)
        else:
            rationales_c1.add(feature)
        
        x = sp.csr_matrix(X_pool[doc_id], dtype=float)
         
        x_feats = x[0].indices
        for f in x_feats:
            if f == feature:
                x[0,f] = w_r*x[0,f]
            else:
                x[0,f] = w_o*x[0,f]
        

        
        if not y_train:
            X_train = x
        else:
            X_train = sp.vstack((X_train, x))
        
        y_train.append(y_pool[doc_id])
    
    # Train the model
    
    if model_type=='lrl2':
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)        
    elif model_type=='mnb':        
        model = MultinomialNB(alpha=1)        
    elif model_type=='svm_linear':
        model = svm.SVC(kernel='linear', C=svm_C, probability=True)
        #model = svm.SVC(kernel='linear', probability=True)        
    elif model_type=='svm_rbf':
        model = svm.SVC(kernel='rbf', gamma=svm_gamma, C=svm_C, probability=True)
        #model = svm.SVC(kernel='rbf', probability=True)        
    elif model_type=='svm_poly':
        model = svm.SVC(kernel='rbf', gamma=svm_gamma, C=svm_C, probability=True)
        #model = svm.SVC(kernel='poly', probability=True)        
    elif model_type=='adaptive_lr':
        random_state = np.random.RandomState(seed=seed)
        #model = LogisticRegression(C=C, penalty='l2', random_state=random_state)
        #model.fit(X_train, np.array(y_train))
        model = LogisticRegressionAdaptive()   
    elif model_type=='adaptive_svm':
        random_state = np.random.RandomState(seed=seed)        
        model = AdaptiveSVM()      
    elif model_type=='SGD':
            model = SGDClassifier(loss="log",penalty='l2', n_iter=100) 
        
    model.fit(X_train, np.array(y_train))
            
    (accu, auc) = evaluate_model(model, X_test, y_test)
    model_scores['auc'].append(auc)
    model_scores['accu'].append(accu)
    
    num_training_samples.append(X_train.shape[0])
    
    feature_expert.rg.seed(seed)        
    
    if selection_strategy == 'random':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'uncertainty':
        doc_pick_model = UNCSampling(model, feature_expert, y_pool, Debug)        
    elif selection_strategy == "optauc":
        doc_pick_model = OptimizeAUC(X_test, y_test, feature_expert, \
            optimize="R", seed=seed, Debug=Debug)          
    elif selection_strategy == "unc_prefer_no_conflict":
        doc_pick_model = UNCPreferNoConflict(model)
    elif selection_strategy == "unc_prefer_conflict":
        doc_pick_model = UNCPreferConflict(model)
    elif selection_strategy == "unc_three_types":
        doc_pick_model = UNCThreeTypes(model)
    else:
        raise ValueError('Selection strategy: \'%s\' invalid!' % selection_strategy)
 
  
    k = step_size  

    while X_train.shape[0] < budget:                

        # Choose a document based on the strategy chosen
        if selection_strategy == "unc_prefer_no_conflict":
            doc_ids = doc_pick_model.choices(X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "unc_prefer_conflict":
            doc_ids = doc_pick_model.choices(X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "unc_three_types":
            doc_ids = doc_pick_model.choices(X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        else:
            doc_ids = doc_pick_model.choices(X_pool, pool_set, k)
        
        if doc_ids is None or len(doc_ids) == 0:
            break        
        
        for doc_id in doc_ids:
            # Remove the chosen document from pool and add it to the training set
            pool_set.remove(doc_id)
            training_set.append(doc_id)

            #feature = feature_expert.most_informative_feature(X_pool[doc_id], y_pool[doc_id])
            feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])
            
            rationales.add(feature)

            if y_pool[doc_id] == 0:
                rationales_c0.add(feature)
            else:
                rationales_c1.add(feature)
            
            x = sp.csr_matrix(X_pool[doc_id], dtype=float)
            
            x_feats = x[0].indices
            for f in x_feats:
                if f == feature:
                    x[0,f] = w_r*x[0,f]
                else:
                    x[0,f] = w_o*x[0,f]
            

            X_train = sp.vstack((X_train, x))
            y_train.append(y_pool[doc_id])
        
        # Train the model

        
        if model_type=='lrl2':
            random_state2 = np.random.RandomState(seed=seed)        
            model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)                  
        elif model_type=='mnb':        
            model = MultinomialNB(alpha=1)            
        elif model_type=='svm_linear':
            model = svm.SVC(kernel='linear', C=svm_C, probability=True)
            #model = svm.SVC(kernel='linear', probability=True)            
        elif model_type=='svm_rbf':
            model = svm.SVC(kernel='rbf', gamma=svm_gamma, C=svm_C, probability=True)
            #model = svm.SVC(kernel='rbf', probability=True)            
        elif model_type=='svm_poly':
            model = svm.SVC(kernel='rbf', gamma=svm_gamma, C=svm_C, probability=True)
            #model = svm.SVC(kernel='poly', probability=True)   
        elif model_type=='adaptive_lr':
            #random_state = np.random.RandomState(seed=seed)            
            model = LogisticRegressionAdaptive()  
        elif model_type=='adaptive_svm':
            random_state = np.random.RandomState(seed=seed)        
            model = AdaptiveSVM()   
        elif model_type=='SGD':
            model = SGDClassifier(loss="log",penalty='l2', n_iter=100)              
                              

        # fit the model and evaluate
        model.fit(X_train, np.array(y_train))
            
        (accu, auc) = evaluate_model(model, X_test, y_test)
        model_scores['auc'].append(auc)
        model_scores['accu'].append(accu)
        
        num_training_samples.append(X_train.shape[0])
        
  
    print 'Active Learning took %2.2fs' % (time() - start)
    
    return (np.array(num_training_samples), model_scores)

def load_dataset(dataset):
    if dataset == ['imdb']:
        #(X_pool, y_pool, X_test, y_test) = load_data()
        #vect = CountVectorizer(min_df=0.005, max_df=1./3, binary=True, ngram_range=(1,1))
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))        
        X_pool, y_pool, X_test, y_test, _, _, = load_imdb(path='./aclImdb/', shuffle=True, vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif isinstance(dataset, list) and len(dataset) == 3 and dataset[0] == '20newsgroups':
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
        X_pool, y_pool, X_test, y_test, _, _ = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif dataset == ['SRAA']:
        X_pool = pickle.load(open('./SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('./SRAA_y_train.pickle', 'rb'))
        X_test = pickle.load(open('./SRAA_X_test.pickle', 'rb'))
        y_test = pickle.load(open('./SRAA_y_test.pickle', 'rb'))
        feat_names = pickle.load(open('./SRAA_feature_names.pickle', 'rb'))
        return (X_pool, y_pool, X_test, y_test, feat_names)
    elif dataset == ['nova']:
        (X_pool, y_pool, X_test, y_test) = load_nova()
        return (X_pool, y_pool, X_test, y_test, None)
    elif dataset == ['ibnsina']:
        (X_pool, y_pool, X_test, y_test) = load_ibnsina()
        return (X_pool, y_pool, X_test, y_test, None)
    elif dataset == ['creditg']:
        (X_pool, y_pool, X_test, y_test) = load_creditg()
        return (X_pool, y_pool, X_test, y_test, None)
    
def run_trials(model_type, num_trials, dataset, selection_strategy, metric, C, alpha, \
                bootstrap_size, balance, budget, step_size, topk, w_o, w_r, seed=0, lr_C=1, svm_C=1, svm_gamma=0, Debug=False):
    
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
        
        training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)

        # In order to get the best parameters
        if 0:
            # Train classifier
            #
            # For an initial search, a logarithmic grid with basis
            # 10 is often helpful. Using a basis of 2, a finer
            # tuning can be achieved but at a much higher cost.

            C_range = 10.0 ** np.arange(-5, 9)
            gamma_range = 10.0 ** np.arange(-5, 5)
            param_grid = dict(gamma=gamma_range, C=C_range)
            cv = StratifiedKFold(y=y_pool, n_folds=5)
            grid = GridSearchCV(SVC(kernel='poly'), param_grid=param_grid, cv=cv)
            grid.fit(X_pool, np.array(y_pool))
            print("The best classifier is: ", grid.best_estimator_)

            # Now we need to fit a classifier for all parameters in the 2d version
            # (we use a smaller set of parameters here because it takes a while to train)
            C_2d_range = [1, 1e2, 1e4]
            gamma_2d_range = [1e-1, 1, 1e1]
            classifiers = []
            for C in C_2d_range:
                for gamma in gamma_2d_range:
                    clf = SVC(C=C, gamma=gamma)
                    clf.fit(X_pool, np.array(y_pool))
                    classifiers.append((C, gamma, clf))
                
        result[i] = learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, fe, \
                          selection_strategy, budget, step_size, topk, w_o, w_r, trial_seed, lr_C, svm_C, svm_gamma, Debug)
    
    return result, feat_names, feat_freq

def average_results(result):
    avg_M_scores = dict()
    
    num_trials = result.shape[0]
    
    if num_trials == 1:
        num_training_set, M_scores = result[0]
        return np.array([(num_training_set, M_scores)])
           
    min_training_samples = np.inf
    for i in range(num_trials):
        # result[i][0] is the num_training_set
        min_training_samples = min(result[i][0].shape[0], min_training_samples)
    
    for i in range(num_trials):
        num_training_set, M_scores = result[i]
        if i == 0:
            avg_M_scores['accu'] = np.array(M_scores['accu'])[:min_training_samples]
            avg_M_scores['auc'] = np.array(M_scores['auc'])[:min_training_samples]
        else:
            avg_M_scores['accu'] += np.array(M_scores['accu'])[:min_training_samples]
            avg_M_scores['auc'] += np.array(M_scores['auc'])[:min_training_samples]
           
    num_training_set = num_training_set[:min_training_samples]
    avg_M_scores['accu'] = avg_M_scores['accu'] / num_trials
    avg_M_scores['auc'] = avg_M_scores['auc'] / num_trials
    
    return np.array([(num_training_set, avg_M_scores)])

def save_result(result, filename='result.txt'):
    # Saves the data the following order:
    # training sample index, IM_accu, FM_accu, PM_accu, IM_acu, FM_auc, PM_auc, c0_features_discovered so far,
    # c1_features_discovered so far, num_docs_covered, and transition phases for cover_then_disagree approach
    # if the approach is not cover_then_disagree, no transition is saved
    print '-' * 50
    print 'saving result into \'%s\'' % filename
    
    ls_all_results = []
    with open(filename, 'w') as f:
        for i in range(result.shape[0]):
            num_training_set, model_scores = result[i]

            ls_all_results.append(num_training_set)
            ls_all_results.append(model_scores['accu'])
            ls_all_results.append(model_scores['auc'])
            
        header = 'train#\tM_accu\tM_auc'
        f.write('\t'.join([header]*result.shape[0]) + '\n')
        for row in map(None, *ls_all_results):
            f.write('\t'.join([str(item) if item is not None else ' ' for item in row]) + '\n')
       

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
    
    filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, 'w_a={:0.2f}'.format(args.w_r), 'w_n={:0.2f}'.format(args.w_o), "num_a_feat_chosen", 'batch-result.txt'])
    
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
    parser.add_argument('-strategy', choices=['random', 'uncertainty', \
                                              'unc_prefer_no_conflict', 'unc_prefer_conflict', 'unc_three_types'], \
                        default='random', help='Document selection strategy to be used')
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
    parser.add_argument('-w_o', type=float, default=1., help='The weight of all features other than rationales')
    parser.add_argument('-w_r', type=float, default=1., help='The weight of all rationales for a document')
    parser.add_argument('-step_size', type=int, default=1, help='number of documents to label at each iteration')
    parser.add_argument('-topk_unc', type=int, default=20, help='number of uncertain documents to consider to differentiate between types of uncertainties')
    parser.add_argument('-model_type', choices=['lrl2', 'adaptive_lr', 'mnb', 'svm_linear', 'svm_rbf', 'svm_poly', 'svm_linear_adaptive', 'SGD'], default='lrl2', help='Type of classifier to be used')
    parser.add_argument('-lr_C', type=float, default=1, help='Penalty term for the logistic regression classifier')
    parser.add_argument('-svm_C', type=float, default=1, help='Penalty term for the SVM classifier')
    parser.add_argument('-svm_gamma', type=float, default=1, help='gamma for SVM')

    args = parser.parse_args()
    
    
            

    result, feat_names, feat_freq = run_trials(model_type=args.model_type, num_trials=args.trials, dataset=args.dataset, selection_strategy=args.strategy,\
                metric=args.metric, C=args.c, alpha=args.alpha, \
                bootstrap_size=args.bootstrap, balance=args.balance, \
                budget=args.budget, step_size=args.step_size, topk=args.topk_unc, \
                w_o=args.w_o, w_r=args.w_r, seed=args.seed, lr_C=args.lr_C, svm_C=args.svm_C, svm_gamma=args.svm_gamma, Debug=args.debug)
    
    print result
    
    for res in result:
        nt, per = res
        accu = per['accu']
        auc = per['auc']
        for i in range(len(nt)):
            print "%d\t%0.4f\t%0.4f" %(nt[i], accu[i], auc[i])
    
    #save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:d}trials'.format(args.trials), 'w_a={:0.2f}'.format(args.w_r), 'w_n={:0.2f}'.format(args.w_o), 'batch-result.txt']))
    save_result(average_results(result), filename='_'.join(['lr_C={:5.5f}'.format(args.lr_C), 'SVM_C={:5.5f}'.format(args.svm_C),'_'.join(args.dataset), args.strategy, args.metric, 'w_r={:2.5f}'.format(args.w_r), 'w_o={:2.5f}'.format(args.w_o), 'averaged', 'batch-result.txt']))
    #save_result(average_results(result), filename='_'.join([args.model_type, 'Results\\','lr_C={:5.5f}'.format(args.lr_C), 'SVM_C={:5.5f}'.format(args.svm_C),'_'.join(args.dataset), args.strategy, args.metric, 'w_r={:2.5f}'.format(args.w_r), 'w_o={:2.5f}'.format(args.w_o), 'averaged', 'batch-result.txt']))

    #save_result_num_a_feat_chosen(result, feat_names, feat_freq)

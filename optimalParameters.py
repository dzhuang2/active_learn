'''
optimal parameters.py

This file calculates the optimal parameters based on the current training set
'''

import sys
import os
sys.path.append(os.path.abspath("."))

#import sys
from time import time
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from feature_expert import print_all_features
from sklearn.naive_bayes import MultinomialNB
from models import FeatureMNBUniform, FeatureMNBWeighted, PoolingMNB
from sklearn import metrics

import pickle
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn import svm


def optimalMNBParameters(X_pool, y_pool, all_features):

    grid_wr=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    grid_wo=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_w_r=-1
    optimal_w_o=-1

    for w_r in grid_wr:    
        
        for w_o in grid_wo:
             
            if w_r>=w_o:   

                all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
                kf = KFold(len(y_pool), n_folds=5)
        
                for train, test in kf:            

                    #features = list(np.array(all_features)[train])
                    # multiply instances in each fold of training data by wr and wo and calculate probabilities for test instances

                    X_train = None
                    y_train = []

                    for doc_id in train:
                        x = sp.csr_matrix(X_pool[doc_id], dtype=float)                
                        x_feats = x[0].indices
                        for f in x_feats:
                            if f == all_features[doc_id]:
                                x[0,f] = w_r*x[0,f]
                            else:
                                x[0,f] = w_o*x[0,f]
                
                        if not y_train:
                            X_train = x
                        else:
                            X_train = sp.vstack((X_train, x))
        
                        y_train.append(y_pool[doc_id])
    
            
                    # alpha=1 for MNB
                    model = MultinomialNB(alpha=1.)       
                    model.fit(X_train, np.array(y_train))

                    X_test=X_pool[test]
                    y_test=y_pool[test]

                    y_probas = model.predict_proba(X_test)            
            
                    counter=0
                    for t in test:
                        all_probabilities[t]=y_probas[counter]
                        counter=counter+1
        
                # compute AUC based on all instances in the training data
                auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
                if auc > max_auc:
                    max_auc = auc
                    optimal_w_r = w_r
                    optimal_w_o = w_o
        


    return optimal_w_r, optimal_w_o


def optimalMNBLwoRParameters(X_pool, y_pool, all_features):

    grid_wr=[0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_w_r=-1

    for w_r in grid_wr:    
        #all_probabilities=np.zeros(len(y_pool))
        all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
        kf = KFold(len(y_pool), n_folds=5)
        
        for train, test in kf:            

            #features = list(np.array(all_features)[train])
            # multiply instances in each fold of training data by wr and wo and calculate probabilities for test instances

            X_train = None
            y_train = []

            for doc_id in train:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)                
                x_feats = x[0].indices
                for f in x_feats:
                    if f == all_features[doc_id]:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_r*x[0,f]
                
                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])
    
            
            # alpha=1 for MNB
            model = MultinomialNB(alpha=1.)       
            model.fit(X_train, np.array(y_train))

            X_test=X_pool[test]
            y_test=y_pool[test]

            y_probas = model.predict_proba(X_test)            
            
            counter=0
            for t in test:
                all_probabilities[t]=y_probas[counter]
                counter=counter+1
        
                
        auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
        if auc > max_auc:
            max_auc = auc
            optimal_w_r = w_r
        


    return optimal_w_r, optimal_w_r


def optimalSVMLwoRParameters(X_pool, y_pool, all_features, seed):

    grid_wr=[0.01, 0.1, 1.0, 10.0, 100.0]
    grid_C=[0.1, 1.0, 10.0]
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_w_r=-1
    optimal_C=-1

    for C in grid_C:    
        
        for w_r in grid_wr:                         

            all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
            kf = KFold(len(y_pool), n_folds=5)
        
            for train, test in kf:                            

                X_train = None
                y_train = []

                for doc_id in train:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)                
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == all_features[doc_id]:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_r*x[0,f]
                
                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])
    
                            
                random_state = np.random.RandomState(seed=seed)
                model = LinearSVC(C=C, random_state=random_state)
                model.fit(X_train, np.array(y_train))

                X_test=X_pool[test]
                y_test=y_pool[test]

                y_decision = model.decision_function(X_test)
            
                counter=0
                for t in test:
                    all_probabilities[t]=y_decision[counter]
                    counter=counter+1
        
            # compute AUC based on all instances in the training data
            auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
            if auc > max_auc:
                max_auc = auc
                optimal_w_r = w_r
                optimal_C = C        


    return optimal_w_r, optimal_w_r, optimal_C
   

def optimalSVMParameters(X_pool, y_pool, all_features, seed):

    grid_wr=[0.01, 0.1, 1.0, 10.0, 100.0]
    grid_wo=[0.01, 0.1, 1.0, 10.0, 100.0]
    grid_C=[0.1, 1.0, 10.0]
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_w_r=-1
    optimal_w_o=-1
    optimal_C=-1

    for C in grid_C:    
        
        for w_r in grid_wr:         
            
            for w_o in grid_wo:     
                
                if w_r>=w_o:           

                    all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
                    kf = KFold(len(y_pool), n_folds=5)
        
                    for train, test in kf:                            

                        X_train = None
                        y_train = []

                        for doc_id in train:
                            x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)                
                            x_feats = x[0].indices
                            for f in x_feats:
                                if f == all_features[doc_id]:
                                    x[0,f] = w_r*x[0,f]
                                else:
                                    x[0,f] = w_o*x[0,f]
                
                            if not y_train:
                                X_train = x
                            else:
                                X_train = sp.vstack((X_train, x))
        
                            y_train.append(y_pool[doc_id])
    
                            
                        random_state = np.random.RandomState(seed=seed)
                        model = LinearSVC(C=C, random_state=random_state)
                        model.fit(X_train, np.array(y_train))

                        X_test=X_pool[test]
                        y_test=y_pool[test]

                        y_decision = model.decision_function(X_test)
            
                        counter=0
                        for t in test:
                            all_probabilities[t]=y_decision[counter]
                            counter=counter+1
        
                    # compute AUC based on all instances in the training data                                  
                    auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
                    if auc > max_auc:
                        max_auc = auc
                        optimal_w_r = w_r
                        optimal_w_o = w_o
                        optimal_C = C        


    return optimal_w_r, optimal_w_o, optimal_C



def optimalPoolingMNBParameters(X_pool, y_pool, all_features, smoothing, num_feat):

    grid_alpha=[0.01, 0.1, 1.0, 10.0, 100.0]    
    grid_rValue=[100.0, 1000.0]
    grid_IMweight=[0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01]    
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_alpha=-1
    optimal_rValue=-1
    optimal_model_weights=np.zeros(2)
    optimal_model_weights[0]=-1
    optimal_model_weights[1]=-1

    for alpha in grid_alpha:    
        
        for rValue in grid_rValue:         
            
            for IMweight in grid_IMweight:                                      

                all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
                kf = KFold(len(y_pool), n_folds=5)
        
                for train, test in kf:                            

                    X_train = None
                    y_train = []
                    
                    rationales_c0  = set()
                    rationales_c1  = set()

                    poolingMNBWeights=np.zeros(2)

                    #for doc_id in train:
                    #    if y_pool[doc_id] == 0:
                    #        rationales_c0.add(all_features[doc_id])
                    #    else:
                    #        rationales_c1.add(all_features[doc_id])

                    feature_model=FeatureMNBUniform(rationales_c0, rationales_c1, num_feat, smoothing, [0.5, 0.5], rValue)                    

                    for doc_id in train:

                        if all_features[doc_id]:
                            feature_model.fit(all_features[doc_id], y_pool[doc_id])

                        x = sp.csr_matrix(X_pool[doc_id], dtype=float)                                        
                
                        if not y_train:
                            X_train = x
                        else:
                            X_train = sp.vstack((X_train, x))
        
                        y_train.append(y_pool[doc_id])
    
                            
                    instance_model=MultinomialNB(alpha=alpha)                            
                    instance_model.fit(X_train, y_train)

                    model = PoolingMNB()
                    weights=np.zeros(2)
                    poolingMNBWeights[0]=IMweight
                    poolingMNBWeights[1]=1. - IMweight
                    model.fit(instance_model, feature_model, weights=poolingMNBWeights) # train pooling_model

                    X_test=X_pool[test]
                    y_test=y_pool[test]

                    y_probas = model.predict_proba(X_test)                    
            
                    counter=0
                    for t in test:
                        all_probabilities[t]=y_probas[counter]
                        counter=counter+1
        
                # compute AUC based on all instances in the training data                                            
                auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
                if auc > max_auc:
                    max_auc = auc
                    optimal_alpha = alpha
                    optimal_rValue = rValue
                    optimal_model_weights[0] = IMweight                   
                    optimal_model_weights[1] = 1. - IMweight  


    return optimal_alpha, optimal_rValue, optimal_model_weights

def optimalZaidanParameters(X_pool, y_pool, all_features, seed):

    grid_C=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 10.0]
    grid_Ccontrast=[0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 10.0]
    grid_nu=[0.1, 1.0]
    # search same values for wo, because this is learning without rationales

    max_auc=-1
    optimal_C=-1
    optimal_Ccontrast=-1
    optimal_nu=-1

    for zaidan_C in grid_C:    
        
        for zaidan_Ccontrast in grid_Ccontrast:         
            
            for zaidan_nu in grid_nu:                                      

                all_probabilities=np.ndarray(shape=(len(y_pool),2))
        
                kf = KFold(len(y_pool), n_folds=5)
        
                for train, test in kf:                            

                    X_train = None
                    y_train = []
                    sample_weight = []

                    for doc_id in train:
                        x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
                        if all_features[doc_id] is not None:
                            x_pseudo = (X_pool[doc_id]).todense()                            
        
                            # create pseudoinstances based on rationales provided; one pseudoinstance is created for each rationale.
                            x_feats = x[0].indices
        
                            for f in x_feats:
                                if f == all_features[doc_id]:
                                    x_pseudo[0,f] = x[0,f]/zaidan_nu                        
                                else:                                              
                                    x_pseudo[0,f] = 0.0
              
                            x_pseudo=sp.csr_matrix(x_pseudo, dtype=np.float64)
                        
                        if not y_train:
                            X_train = x      
                            if all_features[doc_id] is not None:      
                                X_train = sp.vstack((X_train, x_pseudo))
                        else:
                            X_train = sp.vstack((X_train, x))
                            if all_features[doc_id] is not None:
                                X_train = sp.vstack((X_train, x_pseudo))
        
                        y_train.append(y_pool[doc_id])
                        if all_features[doc_id] is not None:
                            # append y label again for the pseudoinstance created
                            y_train.append(y_pool[doc_id])
        

                        sample_weight.append(zaidan_C)
                        if all_features[doc_id] is not None:
                            # append instance weight=zaidan_Ccontrast for the pseudoinstance created
                            sample_weight.append(zaidan_Ccontrast)   
    
                            
                    random_state = np.random.RandomState(seed=seed)        
                    model = svm.SVC(kernel='linear',C=1.0, random_state=random_state)

                    model.fit(X_train, np.array(y_train), sample_weight=sample_weight)

                    X_test=X_pool[test]
                    y_test=y_pool[test]

                    y_decision = model.decision_function(X_test)                                                                                    
            
                    counter=0
                    for t in test:
                        all_probabilities[t]=y_decision[counter]
                        counter=counter+1
        
                # compute AUC based on all instances in the training data                                           
                auc = metrics.roc_auc_score(y_pool, all_probabilities[:, 1])
            
            
                if auc > max_auc:
                    max_auc = auc
                    optimal_C = zaidan_C
                    optimal_Ccontrast = zaidan_Ccontrast
                    optimal_nu = zaidan_nu


    return optimal_C, optimal_Ccontrast, optimal_nu

import sys
import os
sys.path.append(os.path.abspath("."))

from time import time
import numpy as np
import argparse
import pickle
import scipy.sparse as sp
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from imdb import load_imdb, load_newsgroups, load_nova
from sklearn import metrics
from feature_expert import feature_expert
from selection_strategies_batch_mbilgic import RandomBootstrap, RandomStrategy, UNCSampling, Pipe
from selection_strategies_batch_mbilgic import UNCPreferNoConflict, UNCPreferNoRationale, UNCPreferRationale, UNCPreferConflict, UNCThreeTypes
from sklearn.svm import LinearSVC
from sklearn import svm
from models import FeatureMNBUniform, PoolingMNB
from optimalParameters import optimalMNBParameters, optimalMNBLwoRParameters, optimalSVMParameters, optimalSVMLwoRParameters, optimalPoolingMNBParameters, optimalZaidanParameters



#20newsgroups comp.os.ms-windows.misc comp.sys.ibm.pc.hardware

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')
  

def learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, feature_expert, \
          selection_strategy, budget, step_size, topk, w_o, w_r, seed=0, alpha=1, smoothing=0, poolingMNBWeights=[0.5, 0.5], poolingFM_r=100.0, lr_C=1, svm_C=1, \
          zaidan_C=1, zaidan_Ccontrast=1, zaidan_nu=1, cvTrain=False, Debug=False):
    
    start = time()
    print '-' * 50
    print 'Starting Active Learning...'
    
    _, num_feat = X_pool.shape
    model_scores = {'auc':[], 'accu':[], 'wr':[], 'wo':[], 'alpha':[], 'svm_C':[], 'zaidan_C':[], 'zaidan_Ccontrast':[], 'zaidan_nu':[], 'FMrvalue':[], 'IMweight':[], 'FMweight':[]}
    
    rationales  = set()
    rationales_c0  = set()
    rationales_c1  = set()

    number_of_docs = 0    

    feature_expert.rg.seed(seed)
    
    num_training_samples = []
    
    all_features=[]
    
    # keep all the training data instance ids in docs list    
    
    docs = training_set          
    
    X_train = None
    y_train = []
    sample_weight = []
      

    for doc_id in docs:
        #feature = feature_expert.most_informative_feature(X_pool[doc_id], y_pool[doc_id])
        feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])         
        
        number_of_docs=number_of_docs+1       
        
        # append feature to all_features, even if it is None
        all_features.append(feature)                     

        if feature is not None:
            rationales.add(feature)

            if y_pool[doc_id] == 0:
                rationales_c0.add(feature)
            else:
                rationales_c1.add(feature)

    if cvTrain:
        # get optimal parameters depending on the model_type

        if model_type=='mnb_LwoR':
            w_r, w_o=optimalMNBLwoRParameters(X_pool[training_set], y_pool[training_set], all_features)

            feature_counter=0
            for doc_id in docs:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                x_feats = x[0].indices
                for f in x_feats:
                    if f == all_features[feature_counter]:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_o*x[0,f]
                feature_counter=feature_counter+1
                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])

                
        elif model_type=='mnb':      
            w_r, w_o=optimalMNBParameters(X_pool[training_set], y_pool[training_set], all_features)

            feature_counter=0
            for doc_id in docs:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                x_feats = x[0].indices
                for f in x_feats:
                    if f == all_features[feature_counter]:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_o*x[0,f]
                feature_counter=feature_counter+1
                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])

        
        elif model_type=='svm_linear':                  
            w_r, w_o, C= optimalSVMParameters(X_pool[training_set], y_pool[training_set], all_features, seed)            

            feature_counter=0
            for doc_id in docs:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                x_feats = x[0].indices
                for f in x_feats:
                    if f == all_features[feature_counter]:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_o*x[0,f]
                feature_counter=feature_counter+1
                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])       
                
        elif model_type=='svm_linear_LwoR':                  

            w_r, w_o, C= optimalSVMLwoRParameters(X_pool[training_set], y_pool[training_set], all_features, seed)
            feature_counter=0
            for doc_id in docs:
                x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                x_feats = x[0].indices
                for f in x_feats:
                    if f == all_features[feature_counter]:
                        x[0,f] = w_r*x[0,f]
                    else:
                        x[0,f] = w_o*x[0,f]
                feature_counter=feature_counter+1
                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])                                    
            

        if model_type=='poolingMNB':   
            classpriors=np.zeros(2)            
            classpriors[1]=(np.sum(y_pool[docs])*1.)/(len(docs)*1.)
            classpriors[0]= 1. - classpriors[1]     
            
            alpha, poolingFM_r, poolingMNBWeights = optimalPoolingMNBParameters(X_pool[training_set], y_pool[training_set], all_features, smoothing, num_feat)
            
            feature_model=FeatureMNBUniform(rationales_c0, rationales_c1, num_feat, smoothing, classpriors, poolingFM_r)

            feature_counter=0
            for doc_id in docs:
                if all_features[feature_counter]:
                    # updates feature model with features one at a time
                    feature_model.fit(all_features[feature_counter], y_pool[doc_id])
                feature_counter=feature_counter+1

                x = sp.csr_matrix(X_pool[doc_id], dtype=float)

                if not y_train:
                    X_train = x
                else:
                    X_train = sp.vstack((X_train, x))
        
                y_train.append(y_pool[doc_id])

        if model_type=='Zaidan':

            zaidan_C, zaidan_Ccontrast, zaidan_nu = optimalZaidanParameters(X_pool[training_set], y_pool[training_set], all_features, seed)

            feature_counter=0

            for doc_id in docs:
                x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
                if all_features[feature_counter] is not None:
                    x_pseudo = (X_pool[doc_id]).todense()
                                
                    # create pseudoinstances based on rationales provided; one pseudoinstance is created for each rationale.
                    x_feats = x[0].indices
        
                    for f in x_feats:
                        if f == all_features[feature_counter]:
                            test= x[0,f]
                            x_pseudo[0,f] = x[0,f]/zaidan_nu
                        else:                                              
                            x_pseudo[0,f] = 0.0                          
                    x_pseudo=sp.csr_matrix(x_pseudo, dtype=np.float64)                
        
                if not y_train:
                    X_train = x      
                    if all_features[feature_counter] is not None:      
                        X_train = sp.vstack((X_train, x_pseudo))
                else:
                    X_train = sp.vstack((X_train, x))
                    if all_features[feature_counter] is not None:
                        X_train = sp.vstack((X_train, x_pseudo))
        
                y_train.append(y_pool[doc_id])
                if all_features[feature_counter] is not None:
                    # append y label again for the pseudoinstance created
                    y_train.append(y_pool[doc_id])
        

                sample_weight.append(zaidan_C)
                if all_features[feature_counter] is not None:
                    # append instance weight=zaidan_Ccontrast for the pseudoinstance created
                    sample_weight.append(zaidan_Ccontrast)  

                feature_counter = feature_counter+1
    
    # Train the model
    
    if model_type=='lrl2':
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)
    elif model_type=='lrl1':
        random_state = np.random.RandomState(seed=seed)
        model = LogisticRegression(C=lr_C, penalty='l1', random_state=random_state)        
    elif model_type=='mnb':        
        model = MultinomialNB(alpha=alpha)        
    elif model_type=='mnb_LwoR':        
        model = MultinomialNB(alpha=alpha)  
    elif model_type=='svm_linear_LwoR':        
        random_state = np.random.RandomState(seed=seed)
        model = LinearSVC(C=C, random_state=random_state)
    elif model_type=='svm_linear':
        random_state = np.random.RandomState(seed=seed)
        model = LinearSVC(C=C, random_state=random_state)
    elif model_type=='poolingMNB':
        instance_model=MultinomialNB(alpha=alpha)        
        model = PoolingMNB()
    elif model_type=='Zaidan':
        random_state = np.random.RandomState(seed=seed)        
        model = svm.SVC(kernel='linear', C=1.0, random_state=random_state)
        
    if model_type=='poolingMNB':                        
        instance_model.fit(X_train, y_train)
        model.fit(instance_model, feature_model, weights=poolingMNBWeights) # train pooling_model
    elif model_type=='Zaidan':
        model.fit(X_train, np.array(y_train), sample_weight=sample_weight)
    else:
        model.fit(X_train, np.array(y_train))
    
    
            
    (accu, auc) = evaluate_model(model, X_test, y_test)
    model_scores['auc'].append(auc)
    model_scores['accu'].append(accu)
    if model_type=='poolingMNB':
        model_scores['alpha'].append(alpha)
        model_scores['FMrvalue'].append(poolingFM_r)
        model_scores['IMweight'].append(poolingMNBWeights[0])
        model_scores['FMweight'].append(poolingMNBWeights[1])
    else:
        model_scores['FMrvalue'].append(0.0)
        model_scores['IMweight'].append(0.0)
        model_scores['FMweight'].append(0.0)

    if model_type=='Zaidan':
        model_scores['zaidan_C'].append(zaidan_C)
        model_scores['zaidan_Ccontrast'].append(zaidan_Ccontrast)
        model_scores['zaidan_nu'].append(zaidan_nu)
    else:
        model_scores['zaidan_C'].append(0.0)
        model_scores['zaidan_Ccontrast'].append(0.0)
        model_scores['zaidan_nu'].append(0.0)

    if model_type=='mnb' or model_type=='mnb_LwoR':
        model_scores['alpha'].append(alpha)        
    else:
        model_scores['alpha'].append(0.0)        

    if model_type=='svm_linear' or model_type=='svm_linear_LwoR':
        model_scores['svm_C'].append(C)        
    else:
        model_scores['svm_C'].append(0.0)
        
    if model_type=='svm_linear' or model_type=='svm_linear_LwoR' or model_type=='mnb' or model_type=='mnb_LwoR':
        model_scores['wr'].append(w_r)
        model_scores['wo'].append(w_o)
    else:
        model_scores['wr'].append(0.0)
        model_scores['wo'].append(0.0)
    
    num_training_samples.append(number_of_docs)
    
    feature_expert.rg.seed(seed)        
    
    if selection_strategy == 'random':
        doc_pick_model = RandomStrategy(seed)
    elif selection_strategy == 'unc':
        doc_pick_model = UNCSampling()         
    elif selection_strategy == "pnc":
        doc_pick_model = UNCPreferNoConflict()
    elif selection_strategy == "pnr":
        doc_pick_model = UNCPreferNoRationale()
    elif selection_strategy == "pr":
        doc_pick_model = UNCPreferRationale()
    elif selection_strategy == "pc":
        doc_pick_model = UNCPreferConflict()
    elif selection_strategy == "tt":
        doc_pick_model = UNCThreeTypes()
    elif selection_strategy == "pipe":
        doc_pick_model = Pipe([UNCSampling(), UNCPreferConflict()], [10, 30])
    else:
        raise ValueError('Selection strategy: \'%s\' invalid!' % selection_strategy)
 
  
    k = step_size  


    #while X_train.shape[0] < budget:     
    while number_of_docs < budget:                       

        # Choose a document based on the strategy chosen
        if selection_strategy == "pnc":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "pnr":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "pr":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "pc":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "tt":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        elif selection_strategy == "pipe":
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k, rationales_c0, rationales_c1, topk)
        else:
            doc_ids = doc_pick_model.choices(model, X_pool, pool_set, k)
        
        if doc_ids is None or len(doc_ids) == 0:
            break        

        

        for doc_id in doc_ids:            
            #feature = feature_expert.most_informative_feature(X_pool[doc_id], y_pool[doc_id])
            feature = feature_expert.any_informative_feature(X_pool[doc_id], y_pool[doc_id])

            all_features.append(feature)                
            
            number_of_docs=number_of_docs + 1
        
            if feature is not None:
                rationales.add(feature)

                if y_pool[doc_id] == 0:
                    rationales_c0.add(feature)
                else:
                    rationales_c1.add(feature)

            # Remove the chosen document from pool and add it to the training set            
            pool_set.remove(doc_id)                        
            training_set.append(long(doc_id))               


        if cvTrain:
        # get optimal parameters depending on the model_type

            X_train = None
            y_train = []
            sample_weight = []

            if model_type=='mnb_LwoR':
                if np.mod(number_of_docs,20)==10:
                    w_r, w_o=optimalMNBLwoRParameters(X_pool[training_set], y_pool[training_set], all_features)

                feature_counter=0
                for doc_id in training_set:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == all_features[feature_counter]:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_o*x[0,f]
                    feature_counter=feature_counter+1
                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])

                
            elif model_type=='mnb':      
                if np.mod(number_of_docs,20)==10:
                    w_r, w_o=optimalMNBParameters(X_pool[training_set], y_pool[training_set], all_features)

                feature_counter=0
                for doc_id in training_set:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=float)
                            
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == all_features[feature_counter]:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_o*x[0,f]
                    feature_counter=feature_counter+1
                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])

        
            elif model_type=='svm_linear_LwoR': 
                if np.mod(number_of_docs,20)==10:                 
                    w_r, w_o, C= optimalSVMLwoRParameters(X_pool[training_set], y_pool[training_set], all_features, seed)
                
                feature_counter=0
                for doc_id in training_set:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
                            
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == all_features[feature_counter]:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_o*x[0,f]

                    feature_counter=feature_counter+1

                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])       
                
            elif model_type=='svm_linear':                  
                if np.mod(number_of_docs,20)==10:
                    w_r, w_o, C= optimalSVMParameters(X_pool[training_set], y_pool[training_set], all_features, seed)

                feature_counter=0
                for doc_id in training_set:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)
                            
                    x_feats = x[0].indices
                    for f in x_feats:
                        if f == all_features[feature_counter]:
                            x[0,f] = w_r*x[0,f]
                        else:
                            x[0,f] = w_o*x[0,f]

                    feature_counter=feature_counter+1

                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])                                    
            

            if model_type=='poolingMNB':   
                classpriors=np.zeros(2)            
                classpriors[1]=(np.sum(y_pool[docs])*1.)/(len(docs)*1.)
                classpriors[0]= 1. - classpriors[1]     
                
                if np.mod(number_of_docs,20)==10:
                    alpha, poolingFM_r, poolingMNBWeights = optimalPoolingMNBParameters(X_pool[training_set], y_pool[training_set], all_features, smoothing, num_feat)
            
                feature_model=FeatureMNBUniform(rationales_c0, rationales_c1, num_feat, smoothing, classpriors, poolingFM_r)

                feature_counter=0
                for doc_id in training_set:
                    if all_features[feature_counter]:
                        # updates feature model with features one at a time
                        feature_model.fit(all_features[feature_counter], y_pool[doc_id])
                    feature_counter=feature_counter+1

                    x = sp.csr_matrix(X_pool[doc_id], dtype=float)

                    if not y_train:
                        X_train = x
                    else:
                        X_train = sp.vstack((X_train, x))
        
                    y_train.append(y_pool[doc_id])

            if model_type=='Zaidan':

                if np.mod(number_of_docs,20)==10:
                    zaidan_C, zaidan_Ccontrast, zaidan_nu = optimalZaidanParameters(X_pool[training_set], y_pool[training_set], all_features, seed)
                
                feature_counter=0

                for doc_id in training_set:
                    x = sp.csr_matrix(X_pool[doc_id], dtype=np.float64)

                    if all_features[feature_counter] is not None:
                        x_pseudo = (X_pool[doc_id]).todense()
                                
                        # create pseudoinstances based on rationales provided; one pseudoinstance is created for each rationale.
                        x_feats = x[0].indices
        
                        for f in x_feats:
                            if f == all_features[feature_counter]:
                                test= x[0,f]
                                x_pseudo[0,f] = x[0,f]/zaidan_nu
                            else:                                              
                                x_pseudo[0,f] = 0.0                
                                          
                        x_pseudo=sp.csr_matrix(x_pseudo, dtype=np.float64)                
        
                    if not y_train:
                        X_train = x      
                        if all_features[feature_counter] is not None:      
                            X_train = sp.vstack((X_train, x_pseudo))
                    else:
                        X_train = sp.vstack((X_train, x))
                        if all_features[feature_counter] is not None:
                            X_train = sp.vstack((X_train, x_pseudo))
        
                    y_train.append(y_pool[doc_id])
                    if all_features[feature_counter] is not None:
                        # append y label again for the pseudoinstance created
                        y_train.append(y_pool[doc_id])
        

                    sample_weight.append(zaidan_C)
                    if all_features[feature_counter] is not None:
                        # append instance weight=zaidan_Ccontrast for the pseudoinstance created
                        sample_weight.append(zaidan_Ccontrast)  

                    feature_counter = feature_counter+1
        
        # Train the model

        if model_type=='lrl2':
            random_state = np.random.RandomState(seed=seed)
            model = LogisticRegression(C=lr_C, penalty='l2', random_state=random_state)
        elif model_type=='lrl1':
            random_state = np.random.RandomState(seed=seed)
            model = LogisticRegression(C=lr_C, penalty='l1', random_state=random_state)        
        elif model_type=='mnb':        
            model = MultinomialNB(alpha=alpha)        
        elif model_type=='mnb_LwoR':        
            model = MultinomialNB(alpha=alpha)  
        elif model_type=='svm_linear_LwoR':        
            random_state = np.random.RandomState(seed=seed)
            model = LinearSVC(C=C, random_state=random_state)
        elif model_type=='svm_linear':
            random_state = np.random.RandomState(seed=seed)
            model = LinearSVC(C=C, random_state=random_state)
        elif model_type=='poolingMNB':
            instance_model=MultinomialNB(alpha=alpha)        
            model = PoolingMNB()
        elif model_type=='Zaidan':
            random_state = np.random.RandomState(seed=seed)        
            model = svm.SVC(kernel='linear', C=svm_C, random_state=random_state)
        
        if model_type=='poolingMNB':                        
            instance_model.fit(X_train, y_train)
            model.fit(instance_model, feature_model, weights=poolingMNBWeights) # train pooling_model
        elif model_type=='Zaidan':
            model.fit(X_train, np.array(y_train), sample_weight=sample_weight)
        else:
            model.fit(X_train, np.array(y_train))
        
                    
        (accu, auc) = evaluate_model(model, X_test, y_test)
        model_scores['auc'].append(auc)
        model_scores['accu'].append(accu)
        if model_type=='poolingMNB':
            model_scores['alpha'].append(alpha)
            model_scores['FMrvalue'].append(poolingFM_r)
            model_scores['IMweight'].append(poolingMNBWeights[0])
            model_scores['FMweight'].append(poolingMNBWeights[1])
        else:
            model_scores['FMrvalue'].append(0.0)
            model_scores['IMweight'].append(0.0)
            model_scores['FMweight'].append(0.0)

        if model_type=='Zaidan':
            model_scores['zaidan_C'].append(zaidan_C)
            model_scores['zaidan_Ccontrast'].append(zaidan_Ccontrast)
            model_scores['zaidan_nu'].append(zaidan_nu)
        else:
            model_scores['zaidan_C'].append(0.0)
            model_scores['zaidan_Ccontrast'].append(0.0)
            model_scores['zaidan_nu'].append(0.0)

        if model_type=='mnb' or model_type=='mnb_LwoR':
            model_scores['alpha'].append(alpha)        
        else:
            model_scores['alpha'].append(0.0)        

        if model_type=='svm_linear' or model_type=='svm_linear_LwoR':
            model_scores['svm_C'].append(C)        
        else:
            model_scores['svm_C'].append(0.0)
        
        if model_type=='svm_linear' or model_type=='svm_linear_LwoR' or model_type=='mnb' or model_type=='mnb_LwoR':
            model_scores['wr'].append(w_r)
            model_scores['wo'].append(w_o)
        else:
            model_scores['wr'].append(0.0)
            model_scores['wo'].append(0.0)
        
        num_training_samples.append(number_of_docs)
        
  
    print 'Active Learning took %2.2fs' % (time() - start)
    
    return (np.array(num_training_samples), model_scores)

def load_dataset(dataset):
    if dataset == ['imdb']:
        #(X_pool, y_pool, X_test, y_test) = load_data()
        #vect = CountVectorizer(min_df=0.005, max_df=1./3, binary=True, ngram_range=(1,1))
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))        
        #vect = TfidfVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1,1))        
        X_pool, y_pool, X_test, y_test, _, _, = load_imdb(path='./active_learn/aclImdb/', shuffle=True, vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif isinstance(dataset, list) and len(dataset) == 3 and dataset[0] == '20newsgroups':
        vect = CountVectorizer(min_df=5, max_df=1.0, binary=True, ngram_range=(1, 1))
        X_pool, y_pool, X_test, y_test, _, _ = \
        load_newsgroups(class1=dataset[1], class2=dataset[2], vectorizer=vect)
        return (X_pool, y_pool, X_test, y_test, vect.get_feature_names())
    elif dataset == ['sraa']:
        X_pool = pickle.load(open('./active_learn/SRAA/SRAA_X_train.pickle', 'rb'))
        y_pool = pickle.load(open('./active_learn/SRAA/SRAA_y_train.pickle', 'rb'))
        X_test = pickle.load(open('./active_learn/SRAA/SRAA_X_test.pickle', 'rb'))
        y_test = pickle.load(open('./active_learn/SRAA/SRAA_y_test.pickle', 'rb'))
        feat_names = pickle.load(open('./active_learn/SRAA_feature_names.pickle', 'rb'))
        return (X_pool, y_pool, X_test, y_test, feat_names)
    elif dataset == ['nova']:
        (X_pool, y_pool, X_test, y_test) = load_nova()
        return (X_pool, y_pool, X_test, y_test, None)
       
def run_trials(model_type, num_trials, dataset, tfidf, selection_strategy, metric, C, alpha, smoothing, poolingMNBWeights, poolingFM_r,\
                zaidan_C, zaidan_Ccontrast, zaidan_nu, bootstrap_size, balance, budget, step_size, topk, w_o, w_r, seed=0, lr_C=1, svm_C=1, cvTrain=False, Debug=False):
    
    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset(dataset)

        
    if not feat_names:
        feat_names = np.arange(X_pool.shape[1])
    
    feat_freq = np.diff(X_pool.tocsc().indptr)   
    
    fe = feature_expert(X_pool, y_pool, metric, smoothing=1e-6, C=C, pick_only_top=True)
    
    tfidft = TfidfTransformer()
    
    if tfidf:
        print "Performing tf-idf transformation"
        X_pool = tfidft.fit_transform(X_pool)
        X_test = tfidft.transform(X_test)
    
    #if False:
    #    (X_pool, y_pool, X_test, y_test, feat_names) = load_dataset_tfidf(dataset)
    
    result = np.ndarray(num_trials, dtype=object)
    
    for i in range(num_trials):
        print '-' * 50
        print 'Starting Trial %d of %d...' % (i + 1, num_trials)

        trial_seed = seed + i # initialize the seed for the trial
        
        training_set, pool_set = RandomBootstrap(X_pool, y_pool, bootstrap_size, balance, trial_seed)
                
        result[i] = learn(model_type, X_pool, y_pool, X_test, y_test, training_set, pool_set, fe, \
                          selection_strategy, budget, step_size, topk, w_o, w_r, trial_seed, alpha, smoothing, poolingMNBWeights, poolingFM_r, lr_C, svm_C, zaidan_C, zaidan_Ccontrast, zaidan_nu, cvTrain, Debug)
    
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
            avg_M_scores['wr'] = np.array(M_scores['wr'])[:min_training_samples]
            avg_M_scores['wo'] = np.array(M_scores['wo'])[:min_training_samples]
            avg_M_scores['alpha'] = np.array(M_scores['alpha'])[:min_training_samples]
            avg_M_scores['svm_C'] = np.array(M_scores['svm_C'])[:min_training_samples]
            avg_M_scores['zaidan_C'] = np.array(M_scores['zaidan_C'])[:min_training_samples]
            avg_M_scores['zaidan_Ccontrast'] = np.array(M_scores['zaidan_Ccontrast'])[:min_training_samples]
            avg_M_scores['zaidan_nu'] = np.array(M_scores['zaidan_nu'])[:min_training_samples]
            avg_M_scores['FMrvalue'] = np.array(M_scores['FMrvalue'])[:min_training_samples]
            avg_M_scores['IMweight'] = np.array(M_scores['IMweight'])[:min_training_samples]
            avg_M_scores['FMweight'] = np.array(M_scores['FMweight'])[:min_training_samples]
        else:
            avg_M_scores['accu'] += np.array(M_scores['accu'])[:min_training_samples]
            avg_M_scores['auc'] += np.array(M_scores['auc'])[:min_training_samples]
            avg_M_scores['wr'] += np.array(M_scores['wr'])[:min_training_samples]
            avg_M_scores['wo'] += np.array(M_scores['wo'])[:min_training_samples]
            avg_M_scores['alpha'] += np.array(M_scores['alpha'])[:min_training_samples]
            avg_M_scores['svm_C'] += np.array(M_scores['svm_C'])[:min_training_samples]
            avg_M_scores['zaidan_C'] += np.array(M_scores['zaidan_C'])[:min_training_samples]
            avg_M_scores['zaidan_Ccontrast'] += np.array(M_scores['zaidan_Ccontrast'])[:min_training_samples]
            avg_M_scores['zaidan_nu'] += np.array(M_scores['zaidan_nu'])[:min_training_samples]
            avg_M_scores['FMrvalue'] += np.array(M_scores['FMrvalue'])[:min_training_samples]
            avg_M_scores['IMweight'] += np.array(M_scores['IMweight'])[:min_training_samples]
            avg_M_scores['FMweight'] += np.array(M_scores['FMweight'])[:min_training_samples]
           
    num_training_set = num_training_set[:min_training_samples]
    avg_M_scores['accu'] = avg_M_scores['accu'] / num_trials
    avg_M_scores['auc'] = avg_M_scores['auc'] / num_trials
    avg_M_scores['wr'] = avg_M_scores['wr'] / num_trials
    avg_M_scores['wo'] = avg_M_scores['wo'] / num_trials
    avg_M_scores['alpha'] = avg_M_scores['alpha'] / num_trials
    avg_M_scores['svm_C'] = avg_M_scores['svm_C'] / num_trials
    avg_M_scores['zaidan_C'] = avg_M_scores['zaidan_C'] / num_trials
    avg_M_scores['zaidan_Ccontrast'] = avg_M_scores['zaidan_Ccontrast'] / num_trials
    avg_M_scores['zaidan_nu'] = avg_M_scores['zaidan_nu'] / num_trials
    avg_M_scores['FMrvalue'] = avg_M_scores['FMrvalue'] / num_trials
    avg_M_scores['IMweight'] = avg_M_scores['IMweight'] / num_trials
    avg_M_scores['FMweight'] = avg_M_scores['FMweight'] / num_trials
    
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
            ls_all_results.append(model_scores['wr'])
            ls_all_results.append(model_scores['wo'])
            ls_all_results.append(model_scores['alpha'])
            ls_all_results.append(model_scores['svm_C'])
            ls_all_results.append(model_scores['zaidan_C'])
            ls_all_results.append(model_scores['zaidan_Ccontrast'])
            ls_all_results.append(model_scores['zaidan_nu'])
            ls_all_results.append(model_scores['FMrvalue'])
            ls_all_results.append(model_scores['IMweight'])
            ls_all_results.append(model_scores['FMweight'])
            
        header = 'train#\tM_accu\tM_auc\tw_r\tw_o\talpha\tsvm_C\tzaidan_C\tzaidan_Ccontrast\tzaidan_nu\tFMrvalue\tIMweight\tFMweight'
        f.write('\t'.join([header]*result.shape[0]) + '\n')
        for row in map(None, *ls_all_results):
            f.write('\t'.join([str(item) if item is not None else ' ' for item in row]) + '\n')


def evaluate_model(model, X_test, y_test):        
    if isinstance(model, MultinomialNB):
        y_probas = model.predict_proba(X_test)
        auc = metrics.roc_auc_score(y_test, y_probas[:, 1])
    else: 
        y_decision = model.decision_function(X_test)
        auc = metrics.roc_auc_score(y_test, y_decision)
    if isinstance(model, PoolingMNB):
        pred_y = model.classes_[np.argmax(y_probas, axis=1)]
    else:
        pred_y = model.predict(X_test)
    accu = metrics.accuracy_score(y_test, pred_y)
    return (accu, auc)
    


#def evaluate_model(model, X_test, y_test):
#    # model must have predict_proba, classes_
#    #y_probas = model.predict_proba(X_test)
#    #auc = metrics.roc_auc_score(y_test, y_probas[:, 1])
#    #pred_y = model.classes_[np.argmax(y_probas, axis=1)]
#    if isinstance(model, MultinomialNB):
#        y_probas = model.predict_proba(X_test)
#        auc = metrics.roc_auc_score(y_test, y_probas[:, 1])
#    else: 
#        y_decision = model.decision_function(X_test)
#        auc = metrics.roc_auc_score(y_test, y_decision)
#    pred_y = model.predict(X_test)
#    accu = metrics.accuracy_score(y_test, pred_y)
#    return (accu, auc)
#    #y_pred = model.predict(X_test)
#    #accu = metrics.accuracy_score(y_test, y_pred)
#    #return (accu, 0)

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
    parser.add_argument('-strategy', choices=['random', 'unc', \
                                              'pnc', 'pc', 'pnr', 'pr', 'tt', 'pipe'], \
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
    parser.add_argument('-instance_model_weight', type=float, default=1, help='weight for the PoolingMNB instance model')
    parser.add_argument('-w_o', type=float, default=1., help='The weight of all features other than rationales')
    parser.add_argument('-w_r', type=float, default=1., help='The weight of all rationales for a document')
    parser.add_argument('-step_size', type=int, default=1, help='number of documents to label at each iteration')
    parser.add_argument('-topk_unc', type=int, default=20, help='number of uncertain documents to consider to differentiate between types of uncertainties')
    parser.add_argument('-model_type', choices=['lrl2', 'lrl1', 'mnb', 'mnb_LwoR', 'svm_linear', 'svm_linear_LwoR', 'poolingMNB', 'Zaidan'], default='lrl2', help='Type of classifier to be used')
    parser.add_argument('-lr_C', type=float, default=1, help='Penalty term for the logistic regression classifier')
    parser.add_argument('-svm_C', type=float, default=1, help='Penalty term for the SVM classifier')
    parser.add_argument('-tfidf', default=False, action='store_true', help='Perform tf-idf transformation [default is false]')
    parser.add_argument('-file_tag', default='', help='the additional tag you might want to give to the saved file')
    parser.add_argument('-smoothing', type=float, default=0, help='smoothing parameter for the feature MNB model')
    parser.add_argument('-poolingFM_r', type=float, default=100., help='r parameter for the feature MNB model')
    parser.add_argument('-zaidan_Ccontrast', type=float, default=1, help='Ccontrast for zaidans paper')
    parser.add_argument('-zaidan_C', type=float, default=1, help='C for zaidans paper')
    parser.add_argument('-zaidan_nu', type=float, default=1, help='nu for zaidans paper')    
    parser.add_argument('-cvTrain', default=False, action='store_true', help='Optimize parameters for each model using cross validation on training data [default is false]')

    args = parser.parse_args()
    
    poolingMNBWeights=np.zeros(2)
    
    poolingMNBWeights[0]=args.instance_model_weight
    poolingMNBWeights[1]=1.0-poolingMNBWeights[0]

    result, feat_names, feat_freq = run_trials(model_type=args.model_type, num_trials=args.trials, dataset=args.dataset, tfidf=args.tfidf, selection_strategy=args.strategy,\
                metric=args.metric, C=args.c, alpha=args.alpha, smoothing=args.smoothing, poolingMNBWeights=poolingMNBWeights, poolingFM_r=args.poolingFM_r, \
                zaidan_C=args.zaidan_C, zaidan_Ccontrast=args.zaidan_Ccontrast, zaidan_nu=args.zaidan_nu, bootstrap_size=args.bootstrap, balance=args.balance, \
                budget=args.budget, step_size=args.step_size, topk=args.topk_unc, \
                w_o=args.w_o, w_r=args.w_r, seed=args.seed, lr_C=args.lr_C, svm_C=args.svm_C, cvTrain=args.cvTrain, Debug=args.debug)
    
    print result
    
    for res in result:
        nt, per = res
        accu = per['accu']
        auc = per['auc']
        for i in range(len(nt)):
            print "%d\t%0.4f\t%0.4f" %(nt[i], accu[i], auc[i])
    
    
    save_result(average_results(result), filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), 'cvTrain{}'.format(args.cvTrain), args.file_tag, args.model_type, args.strategy, args.metric, 'averaged', 'batch-result.txt']))
    save_result(result, filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), 'cvTrain{}'.format(args.cvTrain), args.file_tag, args.model_type, args.strategy, args.metric, 'all', 'batch-result.txt']))
    
    #if args.model_type=='poolingMNB':
    #    save_result(average_results(result), filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha), 'pooling_r={:6.1f}'.format(args.poolingFM_r), 'IMweight={:1.3f}'.format(poolingMNBWeights[0]), 'FMweight={:1.3f}'.format(poolingMNBWeights[1]), 'averaged', 'batch-result.txt']))
    #    save_result(result, filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha),'pooling_r={:6.1f}'.format(args.poolingFM_r), 'IMweight={:1.3f}'.format(poolingMNBWeights[0]), 'FMweight={:1.3f}'.format(poolingMNBWeights[1]),'all', 'batch-result.txt']))
    #elif args.model_type=='Zaidan':
    #    save_result(average_results(result), filename='_'.join(['Zaidan', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'Z_C={:5.3f}'.format(args.zaidan_C), 'Z_Ccontrast={:5.3f}'.format(args.zaidan_Ccontrast), 'Z_nu={:5.3f}'.format(args.zaidan_nu),  'averaged', 'batch-result.txt']))
    #    save_result(result, filename='_'.join(['Zaidan', args.dataset[0], 'tfidf{}'.format(args.tfidf), args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'Z_C={:5.3f}'.format(args.zaidan_C), 'Z_Ccontrast={:5.3f}'.format(args.zaidan_Ccontrast), 'Z_nu={:5.3f}'.format(args.zaidan_nu), 'all', 'batch-result.txt']))
    #else:
    #    #save_result(result, filename='_'.join(['_'.join(args.dataset), args.strategy, args.metric, '{:d}trials'.format(args.trials), 'w_a={:0.2f}'.format(args.w_r), 'w_n={:0.2f}'.format(args.w_o), 'batch-result.txt']))
    #    save_result(average_results(result), filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'SVM_C={:5.3f}'.format(args.svm_C), 'w_r={:2.3f}'.format(args.w_r), 'w_o={:2.3f}'.format(args.w_o), 'averaged', 'batch-result.txt']))
    #    save_result(result, filename='_'.join([args.dataset[0], 'tfidf{}'.format(args.tfidf), args.file_tag, args.model_type, args.strategy, args.metric, 'alpha={:5.4f}'.format(args.alpha), 'lr_C={:5.3f}'.format(args.lr_C), 'SVM_C={:5.3f}'.format(args.svm_C), 'w_r={:2.3f}'.format(args.w_r), 'w_o={:2.3f}'.format(args.w_o), 'all', 'batch-result.txt']))



    #save_result(average_results(result), filename='_'.join([args.model_type, 'Results\\','lr_C={:5.5f}'.format(args.lr_C), 'SVM_C={:5.5f}'.format(args.svm_C),'_'.join(args.dataset), args.strategy, args.metric, 'w_r={:2.5f}'.format(args.w_r), 'w_o={:2.5f}'.format(args.w_o), 'averaged', 'batch-result.txt']))    
    #save_result_num_a_feat_chosen(result, feat_names, feat_freq)

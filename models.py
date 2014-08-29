import numpy as np
from sklearn.naive_bayes import MultinomialNB
np.seterr(divide='ignore')

class FeatureMNBUniform(MultinomialNB):
    def __init__(self, class0_features, class1_features, num_feat, smoothing, class_prior = [0.5, 0.5], r=100.):
        self.class0_features = list(class0_features)
        self.class1_features = list(class1_features)
        self.smoothing = smoothing
        self.num_features = num_feat
        self.class_prior = class_prior
        self.r = r
    
    def update(self):
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
        
        if self.class1_features != []:
            self.feature_log_prob_[1][self.class1_features] = np.log(1./(n0+n1)) # Equation 12
            self.feature_log_prob_[0][self.class1_features] = np.log(1./((n0+n1)*self.r)) # Equation 13
        
        #Equation 14
        self.feature_log_prob_[0][unlabeled_features] = np.log((n1*(1-1./self.r))/((n0+n1)*nu))
        self.feature_log_prob_[1][unlabeled_features] = np.log((n0*(1-1./self.r))/((n0+n1)*nu))

        self.class_log_prior_ = np.log(self.class_prior)
        self.classes_ = np.array([0, 1])

    def fit(self, feature, label):
        if label == 0:
            new_class0_features = set(self.class0_features)
            new_class0_features.update([feature])
            self.class0_features = list(new_class0_features)
        else:
            new_class1_features = set(self.class1_features)
            new_class1_features.update([feature])
            self.class1_features = list(new_class1_features)
        
        self.update()

class FeatureMNBWeighted(MultinomialNB):
    
    def __init__(self, num_feat, feat_count = None, imaginary_counts = 1., class_prior = [0.5, 0.5]):
        
        if feat_count is not None:
            self.feature_count_ = np.array(feat_count)
        else:
            self.feature_count_ = np.zeros(shape=(2, num_feat))
        
        self.imaginary_counts = imaginary_counts
        
        self.alpha = imaginary_counts
        
        self.class_prior = class_prior        
        self.class_log_prior_ = np.log(self.class_prior)
        
        self.classes_ = np.array([0, 1])

    def update(self):
        self._update_feature_log_prob()        
         
    def fit(self, feature, label):
                
        self.feature_count_[label, feature] += 1.        
        self.update()


class PoolingMNB(MultinomialNB):
    def fit(self, mnb1, mnb2, weights=[0.5, 0.5]):
        self.feature_log_prob_ = np.log(weights[0]*np.exp(mnb1.feature_log_prob_) + \
                                        weights[1]*np.exp(mnb2.feature_log_prob_))
        self.class_log_prior_ = mnb1.class_log_prior_
        self.classes_ = mnb1.classes_

class ReasoningMNB(MultinomialNB):
    """
    A MultinomialNB implementation where the reason (the annotated feature)
    provided by an expert is weighed more than the remaining features.
    """
    
    def __init__(self, alpha=1.0, num_classes=2):
        self.alpha = alpha
        self.num_classes = num_classes
        self.partial_fit_first_call = True  
         
    def partial_fit(self, x, y, f, w):
        """ Incremental fit on a single object.
        
        Parameters
        ----------
        x: the object.
        y: the object's label
        f: the annotated feature. Can be None
        w: weight for the non-annotated features
        
        Returns
        -------
        self : object
            Returns self.
        """
        if x.shape[0] != 1:
            msg = "x must be a single object. Passed %d objects."
            raise ValueError(msg % (x.shape[0]))
        
        if x.shape[0] != y.shape[0]:
            msg = "x.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (x.shape[0], y.shape[0]))
        
        
        _, n_features = x.shape
        
        if self.partial_fit_first_call:
            self.class_count_ = np.zeros(self.num_classes, dtype=np.float64)
            self.feature_count_ = np.zeros((self.num_classes, n_features),
                                           dtype=np.float64)
            self.partial_fit_first_call = False
        
        self.class_count_[y] += 1
        
        #all features
        self.feature_count_[y] += w*x[0]
        
        #the annotated feature
        if f:
            self.feature_count_[y][f] += (1-w)*x[0,f]
        
        self._update_feature_log_prob()
        self._update_class_log_prior()
        return self
    
    def fit(self, X, y, sample_weight=None, class_prior=None):
        raise NotImplementedError("This class does not allow batch fitting.")
        
        
        

Readme

'active_learn.py' deals with running trials on various datasets.  It uses argparse so python active_learn.py would display the switches needed to run the trials using various parameters.  Its learn function assumes that reasoning is used(meaning that there is instance model, feature model, and pooling model).  It contains plot and other utility functions to load the data.

'selection_strategies.py' implemtns various document selection strategies and bootstrapping. So far it contains bootstrap function, random strategy, uncertainty, covering, disagreement, cover then disagree, and cheating.  The constructor for these classes may require the target class labels, but those are only used in the case to debug the code.  Note currently, the debug code for these functions only work in conjunction with imdb dataset.  I do plan on expanding the debug to incorporate other datasets as well.

'feature_expert.py' deals with the feature_expert, whether mutual information or L1 or L1 by sign of the coeffient.  It contains various helper methods to check that it is working correctly or output its features in files for reviewing later.

'exploration.py' uses the random selection strategy to gage how well a dataset performs and output the results into files.

'plot_result.py' plots the result files into one graph.  If only instance model is used, no_reasoning.plot is called, but if pooling model is used, its own plot is called.  The code so far needs to be modified to match the result filenames to generate the graphs desired.

'models.py' contains the Feature model and Pooling model objects.

'sraa.py' and 'load_sraa.py' are files to load the SRAA dataset.  'sraa.py' saves the X_train, y_train, X_test, y_test, Train_corpus, Test_corpus as pickle files for later use.

'imdb.py' contains the code needed to load imdb data as well as 20newsgroup data. For 20newsgroup, the dataset needs the categories names as specified on the 20newsgroup scikit-learn page.

'covering.py' contains code that investigates how does the features cover files. The agreement parameter = 'agree' specifies class conscious, while agreement='any' runs covering with class agnostic. The label = 'negative' to specify the feature from the class(negative is class0, positive is class1) is to be removed first.

'learning_util.py' contains code that I thought was needed during coding but moved to this file due to deprecation.  NO other file imports this one.

'explore_cover.py' contains the exploratory code for covering analysis.

'FM_explore.py' contains the alternative version of feature_expert which I plan on integrating to feature_expert very soon to keep all the feature_expert stuff in one place.



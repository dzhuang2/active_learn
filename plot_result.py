from active_learn import load_result, plot
import no_reasoning
import sys
import numpy as np
import matplotlib.pyplot as plt

def plot_all_results():
    models = {'instance_model':1, 'feature_model':2, 'pooling_model':3}
    types = {'accu':'Accuracy', 'auc':'AUC'}
    # result_files = ['random_result.txt', \
                    # 'uncertaintyIM_result.txt', \
                    # 'uncertaintyFM_result.txt', \
                    # 'uncertaintyPM_result.txt', \
                    # 'disagreement_result.txt', \
                    # 'covering_result.txt', \
                    # 'covering_fewest_result.txt', \
                    # 'cheating_result.txt']
    result_files = ['uncertaintyPM_L1_result.txt', \
                'covering_fewest_L1_result.txt', \
                'cover_then_disagree_L1_result.txt']
    plot_label = [filename.rstrip('_result.txt') for filename in result_files]
    results = np.ndarray(len(result_files), dtype=object)
    for i in range(len(result_files)):
        filename = result_files[i]
        results[i] = load_result(filename)
    
    for model in models.keys():
        for type in types.keys():
            print 'model: %s, type: %s' % (model, type)
            plot_model(results, models[model], type, plot_label, '# of training samples', types[type])

def plot_explore_results():
    types = {'accu':'Accuracy', 'auc':'AUC'}
    result_files = ['SRAA_MultinomialNB(alpha=1)_result.txt', \
                'SRAA_LogisticRegression(C=0.1, penalty=\'l1\')_result.txt', \
                'SRAA_LogisticRegression(C=1, penalty=\'l1\')_result.txt']
    plot_label = ['MNB', 'LR-C0.1', 'LR-C1']
    results = np.ndarray(len(result_files), dtype=object)
    num_training_set = np.ndarray(len(result_files), dtype=object)
    
    for i in range(len(result_files)):
        filename = result_files[i]
        num_training_set[i], results[i] = no_reasoning.load_result(filename)
    
    for type in types.keys():
        scores = [results[i][type] for i in range(len(result_files))]
        graph(num_training_set, scores, plot_label, '# of training samples', types[type])           
            
def plot_model(data, model, type, plot_label, x_label, y_label):
    num_files = data.shape[0]
    num_training_set = np.ndarray(num_files, dtype=object)
    scores = np.ndarray(num_files, dtype=object)
    for i in range(num_files):
        num_training_set[i] = data[i][0]
        scores[i]=data[i][model][type]
    
    graph(num_training_set, scores, plot_label, x_label, y_label)
    
def graph(num_training_set, scores, label, xLabel, yLabel):
    axes_params = [0.1, 0.1, 0.58, 0.75]
    bbox_anchor_coord=(1.02, 1)
    
    # Plot the results
    fig = plt.figure(1)
    ax = fig.add_axes(axes_params)
    for i in range(len(label)):
        ax.plot(num_training_set[i], scores[i], label=label[i])
    
    ax.legend(bbox_to_anchor=bbox_anchor_coord, loc=2)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title('Active Learning with Reasoning')
    plt.show()

def compare_results(cost=1.2):
    models = {'instance_model':1, 'feature_model':2, 'pooling_model':3}
    types = {'accu':'Accuracy', 'auc':'AUC'}
    result_files = ['random_L1_reasoning_result.txt', \
                    'random_no_reasoning_result.txt', \
                    'uncertainty_no_reasoning_result.txt', \
                    'uncertaintyIM_L1_reasoning_result.txt', \
                    'uncertaintyPM_L1_reasoning_result.txt', \
                    'cover_then_disagree_L1_result.txt']
    plot_label = [filename.rstrip('_result.txt') for filename in result_files]
    results = np.ndarray(len(result_files), dtype=object)
    for i in range(len(result_files)):
        filename = result_files[i]
        if 'no_reasoning' in filename:
            (num_training_set, IM_scores) = no_reasoning.load_result(filename)
            results[i] = (num_training_set, IM_scores)
        else:
            (num_training_set, IM_scores, FM_scores, PM_scores) = load_result(filename)
            results[i] = (num_training_set * cost, PM_scores)
    
    for type in types.keys():
        plot_model(results, 1, type, plot_label, 'Cost={:f}'.format(cost), types[type])
    
if __name__ == '__main__':
    if len(sys.argv) == 2:
        filename = sys.argv[1]
        if 'no_reasoning' or 'MultinomialNB' or 'LogisticRegression' in filename:
            no_reasoning.plot(*no_reasoning.load_result(filename))
        else:
            plot(*load_result(sys.argv[1]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'compare':
        compare_results(cost=float(sys.argv[2]))
    else:
        # plot_all_results()
        plot_explore_results()
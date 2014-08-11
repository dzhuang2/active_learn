'''
learning_util.py

This file hosts the utility functions mostly for debugging purposes.
'''

def check_result(result, model, metric, start, end):
    '''
    Unpack the result data structure and inspect the trial data
    
    For example, to display the instance model's accuracy data from 0 to 100,
    CheckResult(result, 'IM', 'accu', 0, 100)
    '''
    model2num = {'IM':1, 'FM':2, 'PM':3}
    print 'Metric: %s' % metric
    for i in range(result.shape[0]):
       score = result[i][model2num[model]]
       print 'Trial #%d:' % i
       print ','.join(['%0.5f' % n for n in score[metric][start:end+1]])

def title(num_trials, strategy, metric):
    '''
    Generate a graph title for the plot
    '''
    accu_file = '%dtrial_%s_%s_accu.png' % (num_trials, strategy, metric)
    auc_file = '%dtrial_%s_%s_auc.png' % (num_trials, strategy, metric)
    return (accu_file, auc_file)
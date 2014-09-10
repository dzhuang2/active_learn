
import csv
import numpy as np
from scipy import stats

def read_data(file_name, data_name):
    with open(file_name, 'rb') as csvfile:
        file_reader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        header = file_reader.next()

        data_indices = [i for i,x in enumerate(header) if x == data_name ]

        data = [[] for i in range(len(data_indices))]
        
        for row in file_reader:
            for i in range(len(data_indices)):
                data[i].append(row[data_indices[i]])
        
        data = np.array(data, dtype=float)
    return data

if __name__ == '__main__':
    Debug = True
    one_tailed = True
    significance_threshold = 0.05
    
    data_name = "RM_auc"
    
    file1_name = "C:\\Users\\mbilgic\\git\\di_zhuang_active_learn\\20newsgroups_comp.os.ms-windows.misc_comp.sys.ibm.pc.hardware_uncertaintyRM_chi2_10trials_w_a=1.00_w_n=0.01_batch-result.txt"
    
    #unc_prefer_no_conflict_R
    #file2_name = "C:\\Users\\mbilgic\\git\\di_zhuang_active_learn\\20newsgroups_comp.os.ms-windows.misc_comp.sys.ibm.pc.hardware_unc_prefer_no_conflict_R_chi2_10trials_w_a=1.00_w_n=0.01_batch-result.txt"
    #unc_three_types_R
    file2_name = "C:\\Users\\mbilgic\\git\\di_zhuang_active_learn\\20newsgroups_comp.os.ms-windows.misc_comp.sys.ibm.pc.hardware_unc_three_types_R_chi2_10trials_w_a=1.00_w_n=0.01_batch-result.txt"
    
    
    data1 = read_data(file1_name, data_name)
    data2 = read_data(file2_name, data_name)
    
    print data1.shape
  
    num_steps = data1.shape[1]
    
    means = np.zeros(shape=(2, num_steps))
    p_values = np.zeros(num_steps)
    
    # The second is winning, tying, losing
    wins, ties, losses = 0, 0, 0
    
    for s in range(1, num_steps):
        d1 = data1[:,s]
        d2 = data2[:,s]
        means[0,s] = np.mean(d1)
        means[1,s] = np.mean(d2)
        p_values[s] = stats.ttest_rel(d1, d2)[1] # this is two-tailed
        
        if one_tailed:
            p_values[s] /= 2.
        
        if p_values[s] < significance_threshold: # significant
            if means[0,s] > means[1,s]: # a loss
                losses += 1
            elif means[0,s] < means[1,s]: # a win
                wins += 1
        else: # a tie
            ties += 1
            
        if Debug:
            print
            print "Means:\t%0.6f\t%0.6f" %(means[0,s], means[1,s])
            print "p-value:\t%0.2f" % (p_values[s])
    
    print
    print '-' * 50    
    print "Second one"
    print "Wins:\t%d" % wins
    print "Ties:\t%d" % ties
    print "Loses:\t%d" %losses
    
    

            
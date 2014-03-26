import os
import sys
import time

count = 0

for pop_size in [1000]:
        for lim_percentage in [10]:
                for num_epochs in [10,25,50]:
                        for lr in [0.1]:
                                for online_training in [1]:
                                        for unique_training in [1]:
                                                for sample_rate in [0]:
                                                        for hiddens in [45,60,80]:
                                                                for corruption_level in [0.1]:
                                                                        for use_good_strings in [1]:
                                                                                for w in [10,20,50]:
                                                                                        for sample_sd in [0.001,0.0001]:
                                                                                                for no_clusters,k_top in [[5,10],[10,5],[10,10],[20,5],[20,10],[20,0]]:
                                                                                                        for trials in range(0,5):
                                                                                                                os.system('qsub ae_rosenbrockonline_{0}_{1}_20_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}_{11}_{12}_{13}_{14}.job'.format(
                                                                                                                        pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,use_good_strings,w,sample_sd,no_clusters,k_top,trials))
                                                                                                                time.sleep(0.1)
                                                                                                                count += 1
                                                                                                                print "added:",count 
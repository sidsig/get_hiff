import os
import sys
import time

count = 0

for pop_size in [500,700,1000]:
        for lim_percentage in [10,20]:
                for num_epochs in [25,50]:
                        for lr in [0.01,0.001]:
                                for online_training in [1]:
                                        for unique_training in [1]:
                                                for sample_rate in [0]:
                                                        for hiddens in [25,50,100]:
                                                                for corruption_level in [0.1,0.2,0.5,0.8,0.9]:
                                                                        for use_good_strings in [1]:
                                                                                for sample_sd in [0.1,0.01,0.001]:
                                                                                        for trials in range(0,5):
                                                                                                os.system('qsub ae_rosenbrock_{0}_{1}_20_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_20_{10}_{11}.job'.format(
                                                                                                        pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,
                                                                                                        hiddens,corruption_level,use_good_strings,sample_sd,trials))
                                                                                                time.sleep(0.1)
                                                                                                count += 1
                                                                                                print "added:",count 
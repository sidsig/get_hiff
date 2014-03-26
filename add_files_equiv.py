import os
import sys
import time

count = 0

for pop_size in [200,500,700,1000]:
        for lim_percentage in [10,20,30,40,50]:  
                for corruption_level in [0.1,0.2,0.3,0.7,0.8]:
                        for num_epochs in [10,20,30,50]:
                                for lr in [0.1,0.01]:
                                        for unique_training in [0,1]:
                                                for hiddens in [40,60,80]:
                                                        for trial in xrange(5):
                                                                os.system('qsub ae_equiv_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.job'.format(
                                                                pop_size,lim_percentage,trial,corruption_level,num_epochs,lr,unique_training,hiddens))
                                                                time.sleep(0.1)
                                                                count += 1
                                                                print "added:",count                           
                                                                                                                                                             
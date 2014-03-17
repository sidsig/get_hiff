def create_text(command):
        text = ""
        text += "#!/bin/sh\n"
        text += "#$ -cwd              # Set the working directory for the job to the current directory\n"
        text += "#$ -V\n"
        text += "#$ -l h_rt=24:0:0    # Request 24 hour runtime\n"
        text += "#$ -l h_vmem=1G      # Request 256MB RAM\n"
        text += "{0}".format(command)
        return text

for pop_size in [500,600,700,1000]:
        for lim_percentage in [10,20]:
                for num_epochs in [10,25,50]:
                        for lr in [0.1,0.01,0.001]:
                                for online_training in [1]:
                                        for unique_training in [1]:
                                                for sample_rate in [0]:
                                                        for hiddens in [10,25,50,100]:
                                                                for corruption_level in [0.1,0.2,0.5,0.8,0.9]:
                                                                        for use_good_strings in [1]:
                                                                                for trials in range(0,5):
                                                                                        command = "python2.6 rbm_solver.py {0} {1} 20 {2} {3} {4} {5} {6} {7} {8} {9} {10}".format(
                                                                                                pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,use_good_strings,trials)
                                                                                        f_text = create_text(command)
                                                                                        f = open("ae_knapsack_500_{0}_{1}_20_{2}_{3}_{4}_{5}_{6}_{7}_{8}_{9}_{10}.job".format(pop_size,lim_percentage,num_epochs,lr,online_training,unique_training,sample_rate,hiddens,corruption_level,use_good_strings,trials),"w")
                                                                                        f.write(f_text)


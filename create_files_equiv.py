def create_text(command):
        text = ""
        text += "#!/bin/sh\n"
        text += "#$ -cwd              # Set the working directory for the job to the current directory\n"
        text += "#$ -V\n"
        text += "#$ -l h_rt=24:0:0    # Request 24 hour runtime\n"
        text += "#$ -l h_vmem=1.5G      # Request 256MB RAM\n"
        text += "{0}".format(command)
        return text

for pop_size in [200,500,700,1000]:
        for lim_percentage in [10,20,30,40,50]:  
                for corruption_level in [0.1,0.2,0.3,0.7,0.8]:
                        for num_epochs in [10,20,30,50]:
                                for lr in [0.1,0.01]:
                                        for unique_training in [0,1]:
                                                for hiddens in [40,60,80]:
                                                        for trial in xrange(5):
                                                        	command = "python2.6 rbm_solver.py {0} {1} {2} {3} {4} {5} {6} {7}".format(
                                                                pop_size,lim_percentage,trial,corruption_level,num_epochs,lr,unique_training,hiddens))
                                                                f_text = create_text(command)
                                                                f = open("ae_equiv_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}.job".format(
                                                                pop_size,lim_percentage,trial,corruption_level,num_epochs,lr,unique_training,hiddens),"w")
                                                                f.write(f_text)
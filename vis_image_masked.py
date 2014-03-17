import numpy as np
import matplotlib.pylab as plt
its = 10
genome_l = 100
low = 0
high = 100
# plt.figure(figsize=(15,3))
mask = np.array(np.loadtxt("mask.dat"),"b")
for i in range(1,20,1):
	plt.figure(figsize=(20,5))
	plt.subplot(1,7,1)
	plt.title("Iteration: {0}".format(i+1))
	solution = np.array(np.loadtxt("rw_population_{0}.dat".format(i)),"b")^mask
	solution_means = np.mean(solution,axis=0)
	plt.bar([_ for _ in range(0,len(solution.T))],solution_means,color="k")
	plt.ylim([0,1])
	plt.xlim([0,len(solution.T)])
	plt.subplot(1,7,2)
	solution = np.loadtxt("random_sampled_population_{0}.dat".format(i))^mask
	solution_means = np.mean(solution,axis=0)
	plt.bar([_ for _ in range(0,len(solution.T))],solution_means,color="k")
	plt.ylim([0,1])
	plt.xlim([0,len(solution.T)])
	plt.subplot(1,7,3)
	solution = np.loadtxt("training_data_{0}.dat".format(i))^mask
	plt.pcolor(solution,cmap="Greys")
	plt.xlim(0,genome_l)
	plt.subplot(1,7,4)
	solution = np.loadtxt("fitness_training_data_{0}.dat".format(i))
	solution = solution.reshape(solution.shape[0],1)
	plt.pcolor(solution,cmap="Reds")
	plt.clim(low,high)
	# plt.xlim(0,105)
	plt.subplot(1,7,5)
	solution = np.loadtxt("population_{0}.dat".format(i))^mask
	plt.pcolor(solution,cmap="Greys")
	plt.xlim(0,genome_l)
	plt.subplot(1,7,6)
	solution = np.loadtxt("fitness_pop_{0}.dat".format(i))
	solution = solution.reshape(solution.shape[0],1)
	plt.pcolor(solution,cmap="Reds")
	plt.clim(low,high)
	# plt.savefig("top_20_noise_0.5.png")
	# plt.subplot(1,7,7)
	# solution = np.loadtxt("fitness_pop_{0}.dat".format(i))
	# fitnesses = np.loadtxt("fitness_pop_{0}.dat".format(0))
	# fitnesses = fitnesses.reshape((fitnesses.shape[0],1))
	# if i > 0:
	# 	for k in range(1,i+1):
	# 		solution = np.loadtxt("fitness_pop_{0}.dat".format(k))
	# 		fitnesses = np.hstack((fitnesses,solution.reshape((fitnesses.shape[0],1))))

	# plt.plot(fitnesses.T,"o")
	# plt.ylim(0,high)
	# plt.xlim(0,330)
	# - Stop premature convergence via - L1/L2 regularisation, learning rate, epochs, use autoencoder solutions as suggestion only
	# - inject noise at output, inject noise at input, Improve autoencoder
	# - deal with training set in a clever way, islands, other diversity techniques, and all alex's "suggestions"
	# plt.show()
	print "saving figure"
	plt.savefig("iteration_{0}.png".format(i))
	plt.close()


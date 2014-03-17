import numpy as np
import matplotlib.pylab as plt
import cPickle
import denoising_autoencoder
import theano
import distance
import matplotlib.gridspec as gridspec
its = 10
genome_l = 100
low = -50
high = 50
mask = np.array(np.loadtxt("mask.dat"),"b")
total_rows = 17

plt.figure(figsize=(7, 12))
gs = gridspec.GridSpec(total_rows, 2,hspace=0.5)
# plt.subplots_adjust( hspace=0.4 )
for iteration in range(1,20):
	solution = np.array(np.loadtxt("rw_population_{0}.dat".format(iteration)),"b")
	params = cPickle.load(open("best_params_{0}.pickle".format(iteration)))
	n_visible = params[0].shape[0]
	n_hiddens = params[0].shape[1]
	ae = denoising_autoencoder.dA(n_visible=n_visible,n_hidden=n_hiddens,W=params[0],hbias=params[1],vbias=params[2])
	ae.build_dA(corruption_level=0.1)
	sample_func = theano.function([ae.input],ae.sample)
	reconstruct_func = theano.function([ae.input],ae.z)

	input_vector = solution[0]
	# output_probabilites = 

	ax = plt.subplot(gs[0:2,:])

	probabilities = []
	for i in range(0,300):
		input_vector=solution[i]
		r=reconstruct_func([input_vector])[0]
		r[np.where(mask==1)]=1-r[np.where(mask==1)]
		probabilities.append(r)
		plt.plot(r)

	# plt.show()

	# r=reconstruct_func(solution)
	# r[np.where(mask==1)]=1-r[np.where(mask==1)]
	# plt.plot(np.mean(r,axis=0))
	probabilities = np.array(probabilities)
	for k in r:
		plt.plot(np.mean(probabilities,axis=0),"k-",linewidth=2.5)
		plt.plot(np.mean(probabilities,axis=0),"wo")

	plt.ylabel("prob of 1")
	plt.xlabel("allele")
	plt.title("Iteration {0}".format(iteration))
	# plt.show()

	input_vector=solution[0]
	r=reconstruct_func([input_vector])[0]
	r[np.where(mask==1)]=1-r[np.where(mask==1)]
	# plt.plot(r)
	# plt.plot(input_vector^mask,"go")
	# plt.show()

	samples = sample_func([input_vector]*10)
	unmasked_input = input_vector^mask
	unmasked_samples = samples^mask
	for m in unmasked_samples:
		for i,l in enumerate(m):
			if unmasked_input[i] == 1 and l == 0:
				m[i] = -1
			elif unmasked_input[i] == 0 and l == 1:
				m[i] = 1
			else:
				m[i] = 0
	ax = plt.subplot(gs[3,:])
	# gs.update(hspace=0.5)
	plt.plot(r,"k-",linewidth=2.5)
	plt.plot(r,"wo",linewidth=2.5)
	# ax.axis("off")
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)

	ax = plt.subplot(gs[4,:])
	plt.pcolor(np.array([unmasked_input]))
	# plt.title("Example Solution (pre AE)")
	ax.axis("off")
	ax = plt.subplot(gs[5:7,:])
	plt.pcolor(unmasked_samples,cmap="Greys")
	# plt.title("10 Example Solutions sampled through AE")
	# ax = plt.subplot2grid((total_rows,2),(7, 0),colspan=2,rowspan=1)
	# distances = [distance.hamming(unmasked_input,s) for s in samples^mask]
	# plt.hist(distances)


	# ax.axis("off")
	ax.grid(True)
	input_vector=solution[150]
	r=reconstruct_func([input_vector])[0]
	r[np.where(mask==1)]=1-r[np.where(mask==1)]
	# plt.plot(r)
	# plt.plot(input_vector^mask,"go")
	# plt.show()

	samples = sample_func([input_vector]*10)
	unmasked_input = input_vector^mask
	unmasked_samples = samples^mask
	for m in unmasked_samples:
		for i,l in enumerate(m):
			if unmasked_input[i] == 1 and l == 0:
				m[i] = -1
			elif unmasked_input[i] == 0 and l == 1:
				m[i] = 1
			else:
				m[i] = 0

	ax = plt.subplot(gs[7,:])

	plt.plot(r,"k-",linewidth=2.5)
	plt.plot(r,"wo",linewidth=2.5)
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)

	ax = plt.subplot(gs[8,:])
	plt.pcolor(np.array([unmasked_input]))
	# plt.title("Example Solution (pre AE)")
	ax.axis("off")
	ax = plt.subplot(gs[9:11,:])
	plt.pcolor(unmasked_samples,cmap="Greys")
	# plt.title("10 Example Solutions sampled through AE")
	# ax = plt.subplot2grid((total_rows,2),(12, 0),colspan=2,rowspan=1)
	# distances = [distance.hamming(unmasked_input,s) for s in samples^mask]
	# plt.hist(distances)
	# ax.axis("off")
	ax.grid(True)


	input_vector=solution[299]
	r=reconstruct_func([input_vector])[0]
	r[np.where(mask==1)]=1-r[np.where(mask==1)]
	# plt.plot(r)
	# plt.plot(input_vector^mask,"go")
	# plt.show()

	samples = sample_func([input_vector]*10)
	unmasked_input = input_vector^mask
	unmasked_samples = samples^mask
	for m in unmasked_samples:
		for i,l in enumerate(m):
			if unmasked_input[i] == 1 and l == 0:
				m[i] = -1
			elif unmasked_input[i] == 0 and l == 1:
				m[i] = 1
			else:
				m[i] = 0

	ax = plt.subplot(gs[11,:])

	plt.plot(r,"k-",linewidth=2.5)
	plt.plot(r,"wo",linewidth=2.5)
	plt.setp( ax.get_xticklabels(), visible=False)
	plt.setp( ax.get_yticklabels(), visible=False)

	ax = plt.subplot(gs[12,:])
	plt.pcolor(np.array([unmasked_input]))
	# plt.title("Example Solution (pre AE)")
	ax.axis("off")
	ax = plt.subplot(gs[13:15,:])
	plt.pcolor(unmasked_samples,cmap="Greys")
	# plt.title("10 Example Solutions sampled through AE")
	# ax = plt.subplot2grid((total_rows,2),(12, 0),colspan=2,rowspan=1)
	# distances = [distance.hamming(unmasked_input,s) for s in samples^mask]
	# plt.hist(distances)
	# ax.axis("off")
	ax.grid(True)
	reconstructions = sample_func(solution)
	unmasked_reconstructions = reconstructions^mask
	distances_t = []
	distances_c = []
	for s_i,s in enumerate(solution):
		s_unmasked = s^mask
		distance_from_target = np.sum(np.abs(np.array(([1]*50)+([0]*50)) - (s_unmasked)))
		distances_t.append(distance_from_target)
		distance_from_child = np.sum(np.abs(unmasked_reconstructions[s_i] - (s_unmasked)))
		distances_c.append(distance_from_child)

	ax = plt.subplot(gs[15:17,:])
	plt.plot(distances_c,distances_t)





	plt.savefig("all_{0}.png".format(iteration))
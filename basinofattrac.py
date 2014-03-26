from deep_da import Deep_dA
import pdb
import numpy
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
import pylab 
import matplotlib.pyplot as plt

numpy.random.seed(1)

class BasinOfAttraction:
	def __init__(self,):
		self.optimal = numpy.ones((1,20))
		self.train_data = numpy.random.random((20,20))
		self.train_data = (self.train_data > 0.1)*1.

	def train_da(self,data,num_epochs=40,lr=0.05,output_folder="",iteration=0):
		train_data = data
		train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
		sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=lr,
                      num_epochs=num_epochs,save=False,output_folder=output_folder,iteration=iteration)

	def experiments(self,):
		corruption_rates = [float(i)/10 for i in xrange(11)]
		self.diffs = []
		inputs = numpy.random.random((1000,20))
		inputs = (inputs>0.8)*1.
		for corruption_rate in corruption_rates:
			self.dA = Deep_dA(n_visible=20,n_hidden=[15],squared_err=False)
			self.dA.build_dA(corruption_rate)
			self.sample_da = theano.function([self.dA.input],self.dA.sample)
			self.train_da(self.train_data)
			outs = self.sample_da(inputs)
			diff = (self.optimal - outs).sum(axis=1)
			#diff = diff.mean()
			self.diffs.append(diff)
		fig = plt.figure()
		corruption_rates.insert(0,'')
		plt.boxplot(self.diffs,notch=True,)
		plt.xticks(range(0,12),corruption_rates,)
		plt.xlabel('corruption level')
		plt.ylabel('hamming distances')
		fig.savefig('test.png')			


if __name__ == '__main__':
	test = BasinOfAttraction()
	test.experiments()
	
from deep_da import Deep_dA
import pdb
import numpy
import theano
from custom_dataset import SequenceDataset
from optimizers import sgd_optimizer
import pylab 
import matplotlib.pyplot as plt
import itertools

numpy.random.seed(1)

class BasinOfAttraction:
	def __init__(self,):
		self.optimal = numpy.ones((1,6))
		self.train_data = numpy.random.random((20,6))
		self.train_data = (self.train_data > 0.3)*1.
		self.all_genotypes = numpy.array(([list("".join(seq)) for seq in itertools.product("01", repeat=6)]),dtype=numpy.float)
		

	def train_da(self,data,num_epochs=40,lr=0.05,output_folder="",iteration=0):
		train_data = data
		train_set = SequenceDataset(train_data,batch_size=20,number_batches=None)
		sgd_optimizer(self.dA.params,[self.dA.input],self.dA.cost,train_set,lr=lr,
                      num_epochs=num_epochs,save=False,output_folder=output_folder,iteration=iteration)

	def experiments(self,):
		corruption_rate = 0.1
		self.dA = Deep_dA(n_visible=6,n_hidden=[15],squared_err=False)
		self.dA.build_dA(corruption_rate)
		self.dA_output = theano.function([self.dA.input],self.dA.z)
		self.train_da(self.train_data)
		self.out_probs = self.dA_output(self.all_genotypes)

	def tran_matrix(self,):
		self.tran_matrix = numpy.zeros((64,64))
		for i in xrange(self.all_genotypes.shape[0]):
			probs = self.out_probs[i]
			masked_probs = numpy.absolute(self.all_genotypes + probs - 1.)
			self.tran_matrix[:,i] = numpy.cumprod(masked_probs,axis=1)[:,-1]		
			numpy.flipud(self.tran_matrix)		
		plt.pcolor(self.tran_matrix)
		plt.colorbar()
		plt.show()

		


if __name__ == '__main__':
	test = BasinOfAttraction()
	test.experiments()
	test.tran_matrix()
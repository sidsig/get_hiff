import cPickle
import numpy
import theano
import theano.tensor as T
import os
from theano.tensor.shared_randomstreams import RandomStreams
import pdb

class FFNN:
    def __init__(self,n_visible,n_hiddens,):
        print 'Initialising FFNN function'
        self.n_visible = n_visible
        self.n_hiddens = n_hiddens  
        self.numpy_rng = numpy.random.RandomState()
        self.theano_rng = RandomStreams()
        initial_W = numpy.asarray(self.numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (self.n_hiddens + self.n_visible)),
                      high=4 * numpy.sqrt(6. / (self.n_hiddens + self.n_visible)),
                      size=(self.n_visible, self.n_hiddens)),
                      dtype=theano.config.floatX)
        self.Wh = theano.shared(value=initial_W, name='Wh', borrow=True)
        self.bh = theano.shared(value=numpy.zeros(self.n_hiddens,dtype=theano.config.floatX),
                                name='bh', borrow=True)
        initial_W = numpy.asarray(self.numpy_rng.uniform(
                      low=-4 * numpy.sqrt(6. / (self.n_hiddens + 1)),
                      high=4 * numpy.sqrt(6. / (self.n_hiddens + 1)),
                      size=(self.n_hiddens, 1)),
                      dtype=theano.config.floatX)
        self.Wo = theano.shared(value=initial_W, name='Wo', borrow=True)
        self.bo = theano.shared(value=numpy.zeros(self.n_hiddens,dtype=theano.config.floatX),
                                name='bo', borrow=True)
        self.input = T.matrix('x')

    def forward_pass(self,):
        self.hidden_activations = T.nnet.sigmoid(T.dot(self.input,self.Wh) + self.bh)
        self.o = T.nnet.sigmoid(T.dot(self.hidden_activations,self.Wo) + self.bo)

    def build_fprop(self,):
        print 'Building theano function.'
        self.forward_pass()
        self.fprop = theano.function([self.input],self.o)

if __name__ == '__main__':
    n_visible = 10
    n_hiddens = 20
    test = FFNN(n_visible,n_hiddens)
    test.build_fprop()
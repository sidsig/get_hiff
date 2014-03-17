import numpy
import numpy as np
class MAXSAT(object):
    def __init__(self):
        super(MAXSAT, self).__init__()
        self.setInstance(1)

    def setInstance(self, i):
        self.readTestsuite(i)

    def readTestsuite(self,i):
        file = open('benchmarks_maxsat/uf100/uf100-0'+str(i)+'.cnf', 'r')
        dataStr = file.readlines()
        
        dataStr = dataStr[8:-3]
        for i in xrange(len(dataStr)):
            dataStr[i] = dataStr[i].strip().split(' ')[0:3]
        self.data = numpy.array(dataStr,dtype=numpy.float)

    def compFit(self, s):
        sum = 0
        for i in range(self.data.shape[0]):
            neg = self.data[i] > 0
            
            sum += np.logical_not( np.logical_xor( int(s[abs(int(self.data[i,0]))-1]), neg[0])  ) or np.logical_not( np.logical_xor( int(s[abs(int(self.data[i,1]))-1]), neg[1])  ) or np.logical_not( np.logical_xor( int(s[abs(int(self.data[i,2]))-1]), neg[2])  )
        return sum


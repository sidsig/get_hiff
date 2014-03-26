import matplotlib.pyplot as plt
import numpy as np
import pdb

data = [np.random.normal(0, std, 1000) for std in range(1, 6)]
plt.boxplot(data, notch=True,)
pdb.set_trace()

plt.show()
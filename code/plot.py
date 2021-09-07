import numpy as np
import matplotlib.pyplot as plt

values = np.genfromtxt("plot.dat", delimiter=" ", dtype=np.int32, filling_values=0)
values = np.transpose(values)

plt.plot(values[0], values[1])
plt.axis([1, 149, 1, 21590])
plt.xlabel('Neighborhood size')
plt.ylabel('Number of iterations')
plt.savefig('plot.png')
plt.show()


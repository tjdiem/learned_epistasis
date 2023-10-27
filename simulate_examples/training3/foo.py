from processing import *
import numpy as np
import matplotlib.pyplot as plt

i = 13

X = convert_sampling_file("sampled_genotypes/sample_" + str(i))
y = convert_command_file1("../commands/command_" + str(i))

X = np.array(X)
y = np.array(y)

ind = np.argmax(y[0])

X = X.sum(axis=0)
print(X.mean())
print(X.min())
print(X[ind])

print(X.shape)
print(ind)

plt.scatter(range(len(X)), X)
plt.scatter(ind,X[ind])
plt.show()
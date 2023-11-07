from processing import *
import numpy as np
import matplotlib.pyplot as plt

for i in range(200,250):

    X = convert_sampling_file("../../test_training/sampled_genotypes/sample_stronger_" + str(i))
    y = convert_command_file1("../../test_training/commands/command_stronger_" + str(i))

    with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
        string = f.readlines()[0].split(" ")

    print(string)

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
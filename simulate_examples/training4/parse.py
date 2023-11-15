#save list of indices where epistatic strength is smaller
import numpy as np
import random

num_samples = 1000

# l = []
# for i in range(50000):
#     with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
#         s = f.readlines()[0].split()
#         if float(s[-1]) > float(s[-4]):
#             l.append(i)

# with open("smaller_epistatic.txt", "w") as f:
#     f.write(str(l))

l = []
idx = list(range(87500))
random.shuffle(idx)
for i in idx[:5000]:

    sampling_file = "../../test_training/sampled_genotypes/sample_stronger_" + str(i)
    try:
        with open(sampling_file, "r") as f:
            lines = f.readlines()
    except FileNotFoundError:
        continue

    X = [[float(l) for l in line[:-1]] for line in lines]

    X = np.array(X) - 1
    with open("../../test_training/commands/command_stronger_" + str(i), "r") as f:
        s = f.readlines()[0].split()

    X = X.sum(axis=0)

    points = [float(s[6]), float(s[7])]
    inds = [round(num_samples*point - 0.5) for point in points]
    regular_site = round(num_samples*float(s[10]) - 0.5)

    if min(X[inds]) > X[regular_site] + 10:
        l.append(i)


print(len(l))




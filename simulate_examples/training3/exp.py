from processing import *
import numpy as np
import matplotlib.pyplot as plt

sigma = 3

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

arange_1d = np.arange(1000)
arange_2d = arange_1d[:,np.newaxis] + np.zeros((1000,1000)) #probably better way of doing this
# print(arange_2d)
# exit()
N = Normal(arange_2d, arange_1d)

sigma = 4
N2 = Normal(arange_2d, arange_1d)
# print(N)
# exit()

def score(inds_pred, ind_true):
    for i, ind in enumerate(inds_pred):
        if abs(ind[0] - ind_true[0]) < 20 and abs(ind[1] - ind_true[1]) < 20:
            return i
        
    return 15


def test(i, num_min):
    X = convert_sampling_file("../../test_training/sampled_genotypes/sample_stronger_" + str(i))

    X = np.array(X) - 1

    num_chrom, num_samples = X.shape

    # X_mean = X.mean(axis=0)
    # X = X - X_mean
    # X_norm = np.linalg.norm(X, axis=0,ord=2)
    # X = X/X_norm
    # print(X_norm.shape)
    # print(X_mean)
    # exit()

    #X = X @ N

    C = X.T @ X / num_chrom

    #C = C + N @ C @ C @ N / 100

    ####
    #C = C - 0.01 * N @ C @ N
    ####

    # m = np.unravel_index(np.argmin(C), C.shape)
    # print("True min", m)
    # print(C[m])

    # D = N @ C @ N

    #C = 0.1 * C @ C / 1000
    #print( C @ C @ C @ C)
    #C = C + 10 * N @ C @ C @ N/ 1000 + 0.8*N @ C @ C @ C @ C @ N / 1000**2
    # E = N @ C @ C @ N
    # C = C + N @ C @ C @ N / 1000

    C = np.triu(C,k=20)
    #print(0.0001 *N @ C @ C @ C @ C @ N / 1000**2)

    inds_pred = []
    indices = np.argsort(C.reshape(-1))[:num_min]
    for ind in indices:
        ind = np.unravel_index(ind, C.shape)
        inds_pred.append(ind)
        print("Minimum index", ind)
        print(C[ind])

    # C = C - 0.01 * N @ C @ N

    print(C/np.linalg.norm(C, "fro"))
    # C = C @ C 
    print(C/np.linalg.norm(C, "fro"))


    y, _ = convert_command_file5("../../test_training/commands/command_stronger_" + str(i))

    inds = []
    for i in range(len(y)):
        if y[i] == 1.0:
            inds.append(i)

    ind_true = tuple(sorted(inds))

    print("True answer", ind_true)

    print(C[ind_true])


    plt.imshow(C)
    plt.colorbar()
    plt.show()

    return score(inds_pred, ind_true)

if __name__ == "__main__":

    # for _ in range(500):
    #     A = np.random.rand(50,50)
    #     A = A/np.linalg.norm(A,"fro")
    #     A = A.T @ A
    #     print(A)
    #     print()
    # exit()


    num_test = 100
    num_min = 10
    i = 7779


    np.random.seed(177)
    idx = np.random.permutation(50_000)[:num_test]
    total_score = 0
    for i in idx:
        # with open("../../test_training/commands/command_stronger_" + str(i),"r") as f:
        #     print(f.readlines())
        total_score += test(i, num_min)

    print("FINAL SCORE", total_score/num_test)
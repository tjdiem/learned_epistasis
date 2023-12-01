from processing import *
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
#from scipy.stats import binom
from ast import literal_eval
# from sklearn.cluster import KMeans

lam_smoothing = 0.0

sigma = 3

Normal = lambda x, mean: e**(-0.5*((x-mean)/sigma)**2) / (sigma * sqrt(2*pi))

arange_1d = np.arange(1000)
arange_2d = arange_1d[:,np.newaxis] + np.zeros((1000,1000)) #probably better way of doing this
# print(arange_2d)
# exit()
N = Normal(arange_2d, arange_1d)

arange_1d = np.arange(100)
arange_2d = arange_1d[:,np.newaxis] + np.zeros((100,100)) #probably better way of doing this
N2 = Normal(arange_2d, arange_1d)
# print(N)
# exit()

def score(inds_pred, ind_true):
    for i, ind in enumerate(inds_pred):
        if abs(ind[0] - ind_true[0]) < 20 and abs(ind[1] - ind_true[1]) < 20:
            return i
        
    return len(inds_pred) + 5

def p_score(array1,array2):
    N = len(array1)
    C = np.dot(array1,array2)
    C = int(C)
    print(C)
    if C < 0:
        array2 = -array2
        C = -C
    
    p_arr1_pos = (array1 == 1).sum()/N
    p_arr1_neg = (array1 == -1).sum()/N

    p_arr2_pos = (array2 == 1).sum()/N
    p_arr2_neg = (array2 == -1).sum()/N

    
    p_arr1_pos = p_arr1_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr1_neg = p_arr1_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing

    p_arr2_pos = p_arr2_pos * (1 - lam_smoothing) + 0.5 * lam_smoothing
    p_arr2_neg = p_arr2_neg * (1 - lam_smoothing) + 0.5 * lam_smoothing
    

    p1 = p_arr1_pos * p_arr2_pos + p_arr1_neg * p_arr2_neg #probability of positive correlation
    p2 = p_arr1_pos * p_arr2_neg + p_arr1_neg * p_arr2_pos #probability of negative correlation

    p3 = 1 - p1 - p2 # probability of no correlation



    val = 0
    # k1 number of pointwise positive correlations
    # k2 number of pointwise negative corerlations
    # we need k1 - k2 >= C
    for k1 in range(C, N+1):
        for k2 in range(0, min(k1 - C + 1, N - k1 + 1)):
            k3 = N - k1 - k2 #number of pointwise neutral correlations
            val += factorial(N)//(factorial(k1)*factorial(k2)*factorial(k3)) * p1**k1 * p2**k2 * p3**k3


    return val

def test(file_number, num_min):
    sampling_file = "../../test_training/sampled_genotypes/sample_stronger_" + str(file_number)
    with open(sampling_file, "r") as f:
        lines = f.readlines()

    X = [[float(l) for l in line[:-1]] for line in lines]

    X = np.array(X) - 1

    num_chrom, num_samples = X.shape

    S = np.sum(X, axis=0) / num_chrom

    S = S.reshape(num_samples, 1)
    S = S @ S.T

    #S = S + S[:,np.newaxis]

    # print(S.shape)
    # exit()

    # X_mean = X.mean(axis=0)
    # X = X - X_mean
    # X_norm = np.linalg.norm(X, axis=0,ord=2)
    # X = X/X_norm
    # print(X_norm.shape)
    # print(X_mean)
    # exit()

    #X = X @ N

    C = X.T @ X / num_chrom #- 0.1 * S

    #C = C - N @ C @ C @ N / 100

    C = np.triu(C,k=20)
    #print(0.0001 *N @ C @ C @ C @ C @ N / 1000**2)

    indices = np.triu_indices_from(C, k=20)


    ### 320 470 518
    # C = np.ones_like(C)
    # for i in range(num_samples):
    #     for j in range(i+20,num_samples):

    # a = 0
    # b = 0
    # for i in range(443,463):
    #     for j in range(665,685):
    #         C[i,j] = p_score(X[:,i],X[:,j])
    #         print(i,j)
    #         print(C[i,j])
    #         if C[i,j] < 0.001:
    #             print(X[:,i])
    #             print(X[:,j])

    #         b += C[i,j]
    #         a += 1

    # print(b/a)


    # values = C[indices]
    # mean = np.mean(values)
    # std = np.std(values)

    # condition1 = C < mean - 5*std
    # condition2 = S < 100000


    # inds_pred = []
    # indices = np.where(condition1 & condition2)
    # for ind1,ind2 in zip(*indices):
    #     inds_pred.append((ind1,ind2))
    #     # print(ind1, ind2)
    #     # print(X[:,ind1], X[:,ind2])

    # print(len(inds_pred))

    # C = np.ones((num_samples,num_samples))
    # for i in range(num_samples):
    #     for j in range(i+1,num_samples):
    #         C[i,j] = p_score(X[:,i],X[:,j])

    inds_pred = []
    indices = np.argsort(C.reshape(-1))[:num_min]
    for ind in indices:
        ind = np.unravel_index(ind, C.shape)
        # print(X[:,ind[0]])
        # print(X[:,ind[1]])
        inds_pred.append(ind)
        print("Minimum index", ind)
        print(C[ind])
        print(S[ind])

    # threshold = 0.6
    # inds_pred = [(ind1, ind2) for ind1,ind2 in inds_pred if p_score(X[:,ind1],X[:,ind2]) < threshold]

    # inds_pred = sorted(inds_pred, key= lambda i: p_score(X[:,i[0]], X[:,i[1]]))

    with open("../../test_training/commands/command_stronger_" + str(file_number), "r") as f:
        s = f.readlines()[0].split()

    points = [float(s[6]), float(s[7])]
    inds = [round(num_samples*point - 0.5) for point in points]
    regular_site = round(num_samples*float(s[10]) - 0.5)

    ind_true = tuple(sorted(inds))

    print("True answer", ind_true)

    print(C[ind_true])
    print(regular_site)



    # plt.imshow(C)
    # plt.colorbar()
    # plt.show()

    regular1 = tuple(sorted([ind_true[0],regular_site]))
    regular2 = tuple(sorted([ind_true[1],regular_site]))

    print(regular1)
    print(regular2)

    print(ind_true[0])
    print(ind_true[1])
    print(regular_site)

    # print()
    # print("TRUE")
    # for i in range(max(ind_true[0]-10,0),min(ind_true[0]+11,1000)):
    #     print(C[i,ind_true[1]])
    #     print(p_score(X[:,i],X[:,ind_true[1]]))
    # with np.printoptions(threshold=100000,linewidth=92):
    #     print(X[:,ind_true[0]-3:ind_true[0]+3])
    #     print()
    #     print(X[:,ind_true[1]-3:ind_true[1]+3])
    #     print()
    #     print(X[:,regular_site-3:regular_site+3])
    # print()
    # print("REGULAR")
    # for i in range(max(0,regular_site-10),min(regular_site+11, 1000)):
    #     print(C[i,ind_true[1]])
    #     print(p_score(X[:,i],X[:,ind_true[1]]))


    return score(inds_pred, ind_true) < min(score(inds_pred, regular1), score(inds_pred, regular2)), 0

if __name__ == "__main__":

    # a = factorial(100)/factorial(50)

    # print(int(a)) 
    # exit()

    # for _ in range(500):
    #     A = np.random.rand(50,50)
    #     A = A/np.linalg.norm(A,"fro")
    #     A = A.T @ A
    #     print(A)
    #     print()
    # exit()

    # X = np.array([[-1,-1,-1,-1,-1,-1,-1, 0],
    #              [-1,-1,-1,-1,-1,-1, 0, 0],
    #              [ 1, 1, 1, 1, 1, 1, 0, 0]]).T
    
    # X_mean = X.mean(axis=0)
    # print(X_mean)

    # X = X - X_mean

    # print(X.T @ X)

    # exit()


    # array1 = np.array([2,1,1,2,2,2,1,0,2,2,2,2,2,1,2]) - 1
    # array2 = np.array([2,1,2,2,2,1,1,1,2,2,1,2,2,1,2]) - 1
    # array3 = np.array([2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]) - 1

    # print(p_score(array1,array2))
    # exit()




    num_test = 500
    num_min = 10
    i = 7779


    np.random.seed(177)
    with open("smaller_strength.txt","r") as f:
        idx = literal_eval(f.read())

    print(len(idx))
    idx = list(range(87500))
    idx = np.random.permutation(idx)[:num_test]
    # idx = [192]
    total_score = 0
    regular_score = 0
    n = 0
    for i in idx:
        try:
            a, b = test(i,num_min)
            total_score += a
            regular_score += b
            n += 1
        except FileNotFoundError:
            pass

    print("REGULAR SCORE", regular_score/n)
    print("FINAL SCORE", total_score/n)

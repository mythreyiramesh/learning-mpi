import numpy as np
from math import sqrt,ceil

def init_matrices(size1,size2,low=0,up=1):
    if (size1[1] != size2[0]):
        print("Cannot multiply matrices of this size!")
        exit(1)
    A = (np.random.rand(size1[0],size1[1])*(up-low)) + low;
    B = (np.random.rand(size2[0],size2[1])*(up-low)) + low;
    return A,B

def factor_procs(nProcs,sizeA,sizeB):
    # aspect_ratio = float(sizeA[0]/sizeB[1])
    flag = 0 # Keeping track of larger dimension
    if sizeA[0] > sizeB[1]:
        aspect_ratio = sizeA[0]/sizeB[1]
    else:
        aspect_ratio = sizeB[1]/sizeA[0]
        flag = 1
    # print("ar",aspect_ratio)
    f1 = ceil(sqrt(float(nProcs)/aspect_ratio))
    # print("f1",f1)
    # print("nPrcs/ar",sqrt(float(nProcs)/aspect_ratio))
    # let iProcs and jProcs be the splits
    # iProcs*jProcs = nProcs & iProcs/jProcs = aspect_ratio
    # iProcs = int(sqrt(nProcs * aspect_ratio))
    # jProcs = int(sqrt(nProcs / aspect_ratio))
    # iProcs = min(iProcs,jProcs)
    while (nProcs%f1 != 0):
        f1 = f1-1
    f2 = nProcs/f1
    if flag:
        return int(f2),int(f1)
    else:
        return int(f1),int(f2)

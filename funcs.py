import numpy as np
from math import sqrt,floor

def init_input_matrices1(size1,size2,low,up):
    A = (np.random.rand(size1[0],size1[1])*(up-low)) + low;
    B = (np.random.rand(size2[0],size2[1])*(up-low)) + low;
    return A,B

def init_input_matrices(size1,size2,low,up):
    n = 16;
    A = np.arange(n,dtype='d').reshape(sqrt(n),sqrt(n))
    B = np.arange(n,dtype='d').reshape(sqrt(n),sqrt(n))
    return A,B

def get_procs1(C_size,nprocs):
    nprocs_root = floor(sqrt(nprocs))
    iProcs = int(nprocs_root)
    jProcs = int(nprocs_root)
    return iProcs,jProcs

def get_procs(C_size,num):
    ar = C_size[0]/C_size[1];
    min_ratio = 100000; # arbitrary large number
    facts = num;
    flag = 0;
    if (ar>1):
        ar = 1/ar;
        flag = 1;
    # print("need ar",ar)
    # print("min ratio diff",min_ratio)
    for i in range(2,int(sqrt(num))+1):
        if (num%i == 0):
            ar1 = (pow(i,2))/(num);
            # print("ar with i",i,"is",ar1)
            if abs((ar/ar1)-1) < min_ratio:
                min_ratio = abs((ar/ar1)-1);
                facts = i;
                # print("new min ratio",min_ratio,"with",facts,num/facts)
    if flag == 1:
        # print((num/facts),facts)
        return (num/facts),facts
    else:
        # print(facts,(num/facts))
        return facts,(num/facts)

# def mult_block(A,B):
#     A_size = np.shape(A)
#     B_size = np.shape(B)
#     C = np.zeros((A_size[0],B_size[1]),dtype='d')
#     if A_size[1] != B_size[0]:
#         print("Sub-matrices cannot be multiplied")
#         exit(1)
#     else:
#         for i in range(A_size[0]):
#             for j in range(B_size[1]):
#                 for k in range(A_size[1]):
#                     C[i][j] += A[i][k]*B[k][j]
#     return C

# [A,B] = init_input_matrices(4,-100,100)
# C = mult_block(A,B)
# C_Act = A.dot(B)
# print(np.amax(np.abs(C-C_Act)))
# print(C)
# print(C_Act)

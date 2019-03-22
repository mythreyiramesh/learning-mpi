import numpy as np
from math import sqrt,floor

def init_input_matrices(size,low,up):
    A = (np.random.rand(size,size)*(up-low)) + low;
    B = (np.random.rand(size,size)*(up-low)) + low;
    return A,B

def get_procs(C_size,nprocs):
    nprocs_root = floor(sqrt(nprocs))
    iProcs = int(nprocs_root)
    jProcs = int(nprocs_root)
    return iProcs,jProcs

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

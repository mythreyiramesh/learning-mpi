import numpy as np
from math import sqrt,floor

def init_input_matrices(size1,size2,low=0,up=0):
    A = (np.random.rand(size1[0],size1[1])*(up-low)) + low;
    B = (np.random.rand(size2[0],size2[1])*(up-low)) + low;
    return A,B

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

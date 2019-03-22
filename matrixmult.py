from math import sqrt,pow

def factors(num):
  factors = []
  for i in range(1,int(sqrt(num))+1):
    if (num%i == 0):
      factors = factors + [i, (num/i)]
  # print(factors)
  return(sorted(factors))

# print(factors(23))

def aspect_ratio_check(num,ar):
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
        print((num/facts),facts)
    else:
        print(facts,(num/facts))
    return facts

# dim = 16x16 * 16*4 = 16*4
aspect_ratio_check(16,4)

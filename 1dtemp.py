import numpy as np

timesteps = 10;
alpha = 1;
delX = 1;
delT = 1;
const = alpha/delX**2;

domain_size = 10;

xgrid = np.linspace(1,10,domain_size);

old_temp = np.sin(xgrid);
new_temp = np.zeros(domain_size);

# since we are using only the previous and next step

lines = ["["]
for j in range(timesteps):
    print(old_temp)
    lines = lines + [str(old_temp)+";\n"]
    for i in range(1,domain_size-1):
        # print("calculating",i,"from",i-1,"and",i+1)
        new_temp[i] = old_temp[i] + const * (old_temp[i-1]-2*old_temp[i]+old_temp[i+1]);
    old_temp = new_temp;

lines = lines + ["];"]
file = open("outs.txt","w");
file.writelines(lines)
file.close()

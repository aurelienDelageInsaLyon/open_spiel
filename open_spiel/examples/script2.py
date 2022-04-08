from matplotlib import pyplot as plt
import numpy as np


valCfrP1=[]
valCfrP2=[]

with open('logAdvTigerH2Cfr.txt') as fff:
    lines = fff.readlines()
i = 0;
for line in lines:
	line = line.split(" ")
	if i%2==0:
		valCfrP1.append(float(line[1]))
	else:
		valCfrP2.append(float(line[1]))
	i+=1
#plt.scatter(range(1,len(valCfrP1)+1),valCfrP1, label="cfrP1",s=0.5);
#plt.scatter(range(1,len(valCfrP2)+1),valCfrP2, label="cfrP2",s=0.5);
res=[]
for j in range(len(valCfrP2)):
	res.append(valCfrP1[j] - valCfrP2[j])
plt.xlabel('nb_iterations');
plt.ylabel('exploitability');
plt.yscale('log')
plt.legend();
plt.plot(res)
plt.savefig('AdvTigerH2.pdf', format='pdf')  
#plt.show()
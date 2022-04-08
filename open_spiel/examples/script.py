from matplotlib import pyplot as plt
import numpy as np
lines = []

valBoundsMaj = []
valBoundsMin = []

with open('boundsH6.txt') as f:
    lines = f.readlines()

it=0
time=[]
for line in lines:
	if (it%2==0):
		line = line.split(" ")
		if (len(line)>5):
			valBoundsMaj.append(float(line[4]))
			valBoundsMin.append(float(line[6]))
	else:
		line = line.split(" ")
		time.append(float(line[3]))
	it= it+1
print(time)
valpomdpMaj = []
valpomdpMin = []

lines = []

with open('pomdpH6.txt') as ff:
    lines = ff.readlines()

for line in lines:
	line = line.split(" ")
	valpomdpMaj.append(float(line[1])-float(line[3]))
	valpomdpMin.append(float(line[3]))


valCfr=[]
timeCfr=[]

with open('log.txt') as fff:
    lines = fff.readlines()

for line in lines:
	line = line.split(" ")
	valCfr.append(float(line[1]))
	timeCfr.append(float(line[2])*2000)

print(valBoundsMaj)
print(valpomdpMaj)
plt.yscale('log')
#plt.plot(time, valpomdpMin, color = "red", label = "POMDP lower-bound")
plt.plot(time, valpomdpMaj, color = "blue", label = "POMDP upper-bound")
#plt.plot(time, valBoundsMin, ':', color = "red", label = "lower-bound")
#plt.plot(time, valBoundsMaj, ':', color = "blue", label = "upper-bound")
plt.plot(timeCfr,valCfr, color="pink", label="cfr");
plt.xlabel('time');
plt.ylabel('value');
plt.legend();

for x in range(len(valpomdpMaj)):
	if (valpomdpMaj[x] > valBoundsMaj[x] + 0.00001):
		print("bug")
		print(valpomdpMaj[x])
		print(valBoundsMaj[x])

for x in range(len(valpomdpMaj)):
	if (valpomdpMin[x] < valBoundsMin[x] - 0.00001):
		print("bug")
		print(valpomdpMin[x])
		print(valBoundsMin[x])

print(valpomdpMaj>valBoundsMaj)
print(valpomdpMin>valBoundsMin)

plt.savefig('MPH6.pdf')  
#plt.show()
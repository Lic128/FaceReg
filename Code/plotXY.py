import random
import time
import math
import matplotlib.pyplot as plt
def genMat(n):
    ret=[[0 for i in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            ret[i][j]=bool(random.getrandbits(1))
            ret[j][i]=ret[i][j]
    return ret

def getTime(n):
    mat=genMat(n)
    start=time.time()
    findTriangle(mat)
    elapsed=(time.time()-start)
    return elapsed

def findTriangle(mat):
    for i in range(len(mat)):
        for j in range(i+1, len(mat)):
            if(mat[i][j]):
                for k in range(j+1, len(mat)):
                    if(mat[i][k] and mat[j][k]):
                        return True

    return False
n_test=[4,8,16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
X=[2,3,4,5,6,7,8,9,10,11,12,13]
Y=[]
for n in n_test:
    print(getTime(n))
    Y.append(math.log(getTime(n)*1000))
plt.figure()
plt.plot(X, Y, label='Time_1')
plt.show()
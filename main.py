import numpy as np
import kNN
group,lables = kNN.createDataSet()
#print(group)
#print(lables)
#print(kNN.classify0([0,0],group,lables,3))
datingDataMat,datingLables = kNN.file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print(datingLables[0:20])

from numpy import array
import matplotlib
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLables),15.0*array(datingLables))
plt.show()

normMat,ranges,minVals = kNN.autoNorm(datingDataMat)
#print(normMat)
#print(ranges)
#print(minVals)

#print(kNN.datingClassTest())

testVector = kNN.img2vector('testDigits/0_13.txt')
#print(testVector[0,0:31])
#print(testVector[0,32:63])

kNN.handwritingClassTest()

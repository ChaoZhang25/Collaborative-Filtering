import DataCleaning as DC
import time
import math
from sklearn import preprocessing
from scipy.sparse import csr_matrix


begin = time.time()

#读入训练矩阵和测试矩阵
#同时读入它们的指示矩阵（即非零位置都是1）
testMat = DC.load_sparse_csr("TestMatrix.npz")
trainMat = DC.load_sparse_csr("TrainMatrix.npz")
trainIndMat = DC.load_sparse_csr("TrainIndicatorMatrix.npz")
testIndMat = DC.load_sparse_csr("TestIndicatorMatrix.npz")

#计算相似度
normMat = preprocessing.normalize(trainMat)
transMat = normMat.transpose()
simMat = normMat.dot(transMat)

#加权求和的矩阵
weightSim = simMat.dot(trainMat)

#分母求和时去除训练矩阵中值为0的项，因此需要乘以指示矩阵
sumMat = simMat.dot(trainIndMat)

#计算得分矩阵，得到评分矩阵
scoreMat = weightSim / sumMat
scoreMatrix = csr_matrix(scoreMat)

#计算RMSE
def rmse(testMat, scoreMatrix, testIndMat):
    X = scoreMatrix.multiply(testIndMat) - testMat
    RMSE = math.sqrt(1.0 * (X.multiply(X)).sum() / 1719466)
    return RMSE


RMSE = rmse(testMat, scoreMatrix, testIndMat)
print "RMSE is: %f" % RMSE

end = time.time()
print "Total Time Consumed is: %f" % (end - begin)

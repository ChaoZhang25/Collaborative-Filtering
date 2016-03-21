import DataCleaning as DC
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.sparse import linalg
from scipy.sparse import csr_matrix
import time
from numpy import random

###############Parameters#################
k = 80
lamda = 0.01
alpha = 0.001
reduce = 0.5
threshold = 10000
##########################################

####使用梯度下降的方法得到预测的评分矩阵########


def frobenius_norm(mat):
    newMat = mat.multiply(mat)
    f = math.sqrt(newMat.sum())
    return f


def objective_function(A, X, U, Vt):
    J = 0.5 * math.pow(frobenius_norm((A.multiply((X - U * Vt)))), 2) + lamda * math.pow(frobenius_norm(U), 2) + lamda * math.pow(frobenius_norm(Vt), 2)
    return J


def derivative_u(A, X, U, Vt):
    du = (A.multiply((U * Vt - X))) * Vt.transpose() + 2 * lamda * U
    return du


def derivative_vt(A, X, U, Vt):
    dv = (A.multiply((U * Vt - X))).transpose() * U + 2 * lamda * Vt.transpose()
    return dv.transpose()


def rmse(testMat, scoreMatrix, testIndMat):
    rmse_begin = time.time()
    X = scoreMatrix.multiply(testIndMat) - testMat
    RMSE = math.sqrt(1.0 * (X.multiply(X)).sum() / 1719466)
    rmse_end = time.time()
    print "time of calculate RMSE is %f" % (rmse_end - rmse_begin)
    return RMSE


##########################################

begin = time.time()

testMat = DC.load_sparse_csr("TestMatrix.npz")
trainMat = DC.load_sparse_csr("TrainMatrix.npz")
trainIndMat = DC.load_sparse_csr("TrainIndicatorMatrix.npz")
testIndMat = DC.load_sparse_csr("TestIndicatorMatrix.npz")

U = random.random(size=(10000, k))
U = U * 0.0001
Vt = random.random(size=(k, 10000))
Vt = Vt * 0.0001
sparse_U = csr_matrix(U)
sparse_Vt = csr_matrix(Vt)

Last = objective_function(trainIndMat, trainMat, sparse_U, sparse_Vt)
scoreMat = sparse_U * sparse_Vt
print "score"
RMSE = rmse(testMat, scoreMat, testIndMat)
R = []
R.append(RMSE)

##########################################

e = 800000
num_iter = 0
while (e > threshold) | (num_iter < 5):
    num_iter += 1
    tmp_U = sparse_U - alpha * derivative_u(trainIndMat, trainMat, sparse_U, sparse_Vt)
    tmp_Vt = sparse_Vt - alpha * derivative_vt(trainIndMat, trainMat, sparse_U, sparse_Vt)
    J = objective_function(trainIndMat, trainMat, tmp_U, tmp_Vt)
    while J > Last:
        alpha = alpha * reduce
        tmp_U = sparse_U - alpha * derivative_u(trainIndMat, trainMat, sparse_U, sparse_Vt)
        tmp_Vt = sparse_Vt - alpha * derivative_vt(trainIndMat, trainMat, sparse_U, sparse_Vt)
        J = objective_function(trainIndMat, trainMat, tmp_U, tmp_Vt)
        print "step reduced to %f" % alpha
    e = Last - J
    Last = J
    sparse_U = tmp_U
    sparse_Vt = tmp_Vt
    scoreMat = sparse_U * sparse_Vt
    RMSE = rmse(testMat, scoreMat, testIndMat)
    R.append(RMSE)
    print"---------------------------------"
    print "Number of Iteration: %d" %num_iter
    print "RMSE is %f" % RMSE
    print "Last Objective Value is %f" % Last
    print "e is %f" % e
    print"---------------------------------"
    # if num_iter > 20:
    #     break

end = time.time()

print "Total Time Consumed is: "
print (end - begin)
print "k = % d" % k
print "lambda = % f" % lamda

##########################################

plt.plot(R)
plt.ylabel('RMSE')
plt.show()



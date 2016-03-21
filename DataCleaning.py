import numpy as np
from scipy.sparse import csr_matrix


def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


def data_cleaning(file_name_train, Matrix_name, size):
    f = open(file_name_train, 'r')
    try:
        all = f.read()
    finally:
        f.close

    Lines = all.split("\r")

    rowTemp = np.zeros((1, size))
    colTemp = np.zeros((1, size))
    dataTemp = np.zeros((1, size))
    indicatorTemp = np.zeros((1, size))

    row = rowTemp[0]
    row.dtype = np.int
    col = colTemp[0]
    col.dtype = np.int

    data = dataTemp[0]
    indicator = indicatorTemp[0]

    index = 0
    rowIndex = -1
    digit = -1

    while True:
        temp = Lines[index].split()
        if int(temp[0]) != digit:
            rowIndex += 1
            digit = int(temp[0])
        row[index] = rowIndex
        col[index] = int(temp[1])-1
        data[index] = float(temp[2])
        if data[index] > 0:
            indicator[index] = 1
        index += 1
        if index % 100000 == 0:
            print index
        if index > (size-1):
            break

    Matrix = csr_matrix((data, (row, col)), shape=(10000, 10000))
    indicatorMatrix = csr_matrix((indicator, (row, col)), shape=(10000, 10000))

    save_sparse_csr(Matrix_name + "Matrix.npz", Matrix)
    save_sparse_csr(Matrix_name + "IndicatorMatrix.npz", indicatorMatrix)

    print "Cleaning Complete Successfully"

file_name_train = "netflix_train.txt"
file_name_test = "netflix_test.txt"
data_cleaning(file_name_train, "Train", 6897746)
data_cleaning(file_name_test, "Test", 1719466)



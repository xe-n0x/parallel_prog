import numpy as np
import sys


def main():
    matrixA = np.loadtxt("matrixA.txt", skiprows=1)
    matrixB = np.loadtxt("matrixB.txt", skiprows=1)
    matrixAB = np.loadtxt("matrixC.txt", skiprows=1)
    matrixRes = np.dot(matrixA, matrixB)
    if np.array_equal(matrixRes, matrixAB):
        return 1
    return 0



if __name__ == "__main__":
    result = main()
    sys.exit(result)
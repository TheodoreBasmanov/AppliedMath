import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
def multiplycation(startVec, P, steps, eps = 0.00001):
    resVec = np.array(startVec)
    curVec = np.array(startVec)
    quadraticDif = []
    P = np.array(P)
    for i in range(1, steps):
        resVec = np.dot(resVec, P)
        #quadraticDif.append(((startVec - curVec)**2).mean(axis=1))
        quadraticDif.append(norm(resVec-curVec))
        if norm(resVec-curVec) < eps:
            break
        curVec = resVec
    plt.plot(quadraticDif)
    plt.ylabel('Mean Squared Error')
    plt.show()
    return resVec

def analyticSolution(P):
    P = np.array(P)
    dim = P.shape[0]
    q = (P-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)

V = [[0.25, 0.05, 0.25, 0, 0.15, 0.25, 0, 0.05], [0, 0.25, 0.05, 0, 0.05, 0.5, 0.1, 0.05], [0.06, 0, 0.34, 0.4, 0.1, 0.01, 0.07, 0.02], [0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.3], [0.21, 0.09, 0.07, 0.33, 0.04, 0, 0.16, 0.1], [0, 0.26, 0.11, 0.03, 0.31, 0.02, 0.17, 0.1], [0.61, 0, 0, 0.01, 0.02, 0.03, 0.3, 0.03], [0, 0, 0.11, 0.12, 0.13, 0.14, 0.24, 0.26]]
startVec = [0, 0.25, 0.05, 0.1, 0.3, 0.1, 0.01, 0.19]
print(analyticSolution(V))
print(multiplycation(startVec, V, 100))
startVec = [0.1, 0.03, 0.3, 0.33, 0.1, 0.04, 0.05, 0.05]
print(multiplycation(startVec, V, 100))

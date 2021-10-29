import math
import numpy as np
from fractions import Fraction
import copy
import os

def getPivotEl(arr, basis):
    for i in range(len(arr)):
        if len(basis) == 0:
            if arr[i] != 0:
                return i
        else:
            if arr[i] != 0 and i + 1 in basis:
                return i
    return -1
def JordanTransformation(xb, basis):
    for row in range(len(xb)):
        newXb = copy.deepcopy(xb)
        col = getPivotEl(xb[row], basis)
        pivotEl = xb[row][col]
        if pivotEl == 0:
            return xb
        for i in range(len(xb)):
            for j in range(len(xb[i])):
                if j == col:
                    newXb[i][j] = 0
                if i == row:
                    newXb[i][j] = xb[i][j]/pivotEl
                if j != col and i != row:
                    newXb[i][j] = xb[i][j] - (xb[row][j]*xb[i][col])/pivotEl
        xb = newXb
    return xb
                

def toTableau(c, A, b, basis):
    xb = [eq + [x] for eq, x in zip(A, b)]
    xb = JordanTransformation(xb, basis)
    minusC = [x * (-1) for x in c]
    z = minusC + [0]
    return xb + [z]

def improveable(tableau, maximizing: bool):
    z = tableau[-1]
    if maximizing:
        return any(x < 0 for x in z[:-1])
    else:
        return any(x > 0 for x in z[:-1])

def getPivotPos(tableau, maximizing: bool):
    z = tableau[-1]
    if maximizing:
        column = next(i for i, x in enumerate(z[:-1]) if x < 0)
    else:
        column = next(i for i, x in enumerate(z[:-1]) if x > 0)
    restrictions = []
    for eq in tableau[:-1]:
        el = eq[column]
        restrictions.append(math.inf if el <= 0 else eq[-1] / el)
    row = restrictions.index(min(restrictions))
    return row, column

def pivotStep(tableau, pivotPosition):
    newTableau = [[] for eq in tableau]
    i, j = pivotPosition
    pivotValue = tableau[i][j]
    newTableau[i] = np.array(tableau[i]) / pivotValue
    for eq_i, eq in enumerate(tableau):
        if eq_i != i:
            multiplier = np.array(newTableau[i]) * tableau[eq_i][j]
            newTableau[eq_i] = np.array(tableau[eq_i]) - multiplier
    return newTableau

def isBasic(column):
    return sum(column) == 1 and len([c for c in column if c == 0]) == len(column) - 1

def solve(tableau):
    columns = np.array(tableau).T
    solutions = []
    for column in columns[:-1]:
        solution = 0
        if isBasic(column): 
            oneIndex = column.tolist().index(1)
            solution = columns[-1][oneIndex]
        solutions.append(solution)   
    return solutions

def simplex(c, A, b, basis, maximizing: bool = True):
    tableau = toTableau(c, A, b, basis)
    while improveable(tableau, maximizing):
        pivotPos = getPivotPos(tableau, maximizing)
        k, j = pivotPos
        tableau = pivotStep(tableau, pivotPos)
    return solve(tableau)

def strListToFloatList(str_list):
    n = 0
    while n < len(str_list):
        str_list[n] = float(str_list[n])
        n += 1
    return(str_list)

folder = os.path.dirname(os.path.abspath(__file__))
filePath = os.path.join(folder, 'Test7.txt')
f = open(filePath, "r")
maximizing = bool(int(f.readline()))
numbers = strListToFloatList(f.readline().split())
c = strListToFloatList(f.readline().split())
A = []
for i in range(int(numbers[1])):
    A.append(strListToFloatList(f.readline().split()))
b = strListToFloatList(f.readline().split())
basis = strListToFloatList(f.readline().split())
f.close()
print(simplex(c, A, b, basis, maximizing))
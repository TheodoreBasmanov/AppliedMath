import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def vectorfield(w, t, p):
    p0, p1, p2, p3 = w
    lam, mu = p

    f = [-lam*p0+mu*p1,
         lam*p0-(mu+lam)*p1+mu*p2,
         lam*p1-(mu+lam)*p2+mu*p3,
         lam*p2-mu*p3]
    return f

lam = 7.23
mu = 2.7
A=np.zeros((5, 4))
A[:][4]=1
A[0][0]=-lam
A[0][1]=mu
A[1][0]=lam
A[1][1]=-(mu+lam)
A[1][2]=mu
A[2][1]=lam
A[2][2]=-(mu+lam)
A[2][3]=mu
A[3][2]=lam
A[3][3]=-mu
b=np.zeros(5)
b[4]=1
print(A)
print(b)
print(np.linalg.lstsq(A,b))

p0 = 1
p1 = 0
p2 = 0
p3 = 0

abserr = 1.0e-8
relerr = 1.0e-6
stoptime = 3.0
numpoints = 250

t = [stoptime * float(i) / (numpoints - 1) for i in range(numpoints)]

p = [lam, mu]
w0 = [p0, p1, p2, p3]

wsol = odeint(vectorfield, w0, t, args=(p,),
              atol=abserr, rtol=relerr)
t_graph = []
p0_graph = []
p1_graph = []
p2_graph = []
p3_graph = []
kbusy_graph = []
kfree_graph = []
for t1, w1 in zip(t, wsol):
        #print (t1, w1[0], w1[1], w1[2], w1[3])
        t_graph.append(t1)
        p0_graph.append(w1[0])
        p1_graph.append(w1[1])
        p2_graph.append(w1[2])
        p3_graph.append(w1[3])
        kbusy_graph.append(w1[1]+w1[2]+w1[3])
        kfree_graph.append(w1[0])

plt.plot(t_graph, p0_graph, label = "p0")
plt.plot(t_graph, p1_graph, label = "p1")
plt.plot(t_graph, p2_graph, label = "p2")
plt.plot(t_graph, p3_graph, label = "p3")
plt.ylabel('Probabilities')
plt.xlabel('Time')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(t_graph, kbusy_graph, label = "busy coefficient")
plt.plot(t_graph, kfree_graph, label = "free coefficient")
plt.ylabel('Coefficients')
plt.xlabel('Time')
plt.legend()
plt.grid(True)
plt.show()
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


# the pdf of two-dimentional normal distribution
def pdf(x, y, rho):
    f = (2 * math.pi * (1 - rho ** 2) ** 0.5) ** -1 * math.exp(
        -1 / (2 * (1 - rho ** 2)) * (x ** 2 + y ** 2 - 2 * rho * x * y))
    return f


# the pdf of normal copula
def pcopula(u, v):
    p = pdf(norm.ppf(u), norm.ppf(v), 0.5) / (norm.pdf(norm.ppf(u)) * norm.pdf(norm.ppf(v)))
    return p


# plot
u = np.arange(0.01, 1.00, 0.01)
X, Y = np.meshgrid(u, u)

Z = []
for i in range(0, 99):
    Z.append([])
    for j in range(0, 99):
        Z[i].append(pcopula(X[i][j], Y[i][j]))

Z = np.asarray(Z)

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('u', fontsize=15)
ax.set_ylabel('v', fontsize=15)
ax.set_zlabel('C', fontsize=15)
ax.set_xlim(0, 1) 
ax.set_ylim(0, 1) 
ax.set_zlim(0, 15) 
surf = ax.plot_surface(X, Y, Z, cmap='rainbow')
fig.colorbar(surf, shrink=0.3, aspect=10)
plt.title('normal copula pdf' + ', ' + r'$\rho$' + '=0.5', fontsize=15)
plt.show()

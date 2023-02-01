import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm


# 定义二元正态分布的 pdf
def pdf(x, y, rho):
    f = (2 * math.pi * (1 - rho ** 2) ** 0.5) ** -1 * math.exp(
        -1 / (2 * (1 - rho ** 2)) * (x ** 2 + y ** 2 - 2 * rho * x * y))
    return f


# 定义 empirical copula 的 pdf
def pcopula(u, v):
    p = pdf(norm.ppf(u), norm.ppf(v), 0.5) / (norm.pdf(norm.ppf(u)) * norm.pdf(norm.ppf(v)))
    return p


# 作图
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
ax.set_xlim(0, 1)  # X轴，横向向右方向
ax.set_ylim(0, 1)  # Y轴,左向与X,Z轴互为垂直
ax.set_zlim(0, 15)  # 竖向为Z轴
surf = ax.plot_surface(X, Y, Z, cmap='rainbow')
fig.colorbar(surf, shrink=0.3, aspect=10)
plt.title('normal copula pdf' + ', ' + r'$\rho$' + '=0.5', fontsize=15)
plt.show()

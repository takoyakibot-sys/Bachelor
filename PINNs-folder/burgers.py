from turtle import color
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
nmax = 400
M = 0.8
alpha2 = 1 - M**2
Uinf = 0.1
dx = dy = 0.05

xs, xe = -1.0, 2.0
ys, ye = 0.0, 1.0
x_le, x_te = 0.0, 1.0

jmax = int((xe - xs) / dx) + 1
kmax = int((ye - ys) / dy) + 1

j_le = int((x_le - xs) / dx)
j_te = int((x_te - xs) / dx) + 1

x = np.linspace(xs, xe, jmax)
y = np.linspace(ys, ye, kmax)

phi = np.zeros([jmax, kmax])
u = np.zeros([jmax, kmax])
v = np.zeros([jmax, kmax])

dydx = np.array([0.2 * (1.0 - 2.0*x[j]) if j_le <= j < j_te else 0.0 for j in range(jmax)])

X, Y = np.meshgrid(x, y)

residual = np.zeros(nmax)
for n in range(nmax):
    phiold = phi.copy()

    # Boundary conditions
    phi[0, :] = 0.0
    phi[jmax-1, :] = 0.0
    phi[:, kmax-1] = 0.0

    for j in range(jmax):
        phi[j, 0] = phi[j, 1] - dydx[j] * dy

    # Gauss-Seidel method
    for k in range(1, kmax-1):
        for j in range(1, jmax-1):
            phi[j, k] = 1.0 / (2.0 * alpha2 + 2.0) * (
                alpha2 * (phi[j-1, k] + phi[j+1, k]) + phi[j, k-1] + phi[j, k+1]
            )

    residual[n] = np.sqrt(((phi - phiold) ** 2).sum() / (jmax * kmax))

# u (x方向速度成分) の計算
for j in range(1, jmax-2):
    u[j, :] = Uinf * (1.0 + (phi[j+1, :] - phi[j-1, :]) / (2*dx))
u[0, :] = Uinf * (1.0 + (phi[1, :] - phi[0, :]) / dx)
u[-1, :] = Uinf * (1.0 + (phi[-1, :] - phi[-2, :]) / dx)

# v (y方向速度成分) の計算
for k in range(1, kmax-2):
    v[:, k] = Uinf * (phi[:, k+1] - phi[:, k-1]) / (2*dy)
v[:, 0] = Uinf * (phi[:, 1] - phi[:, 0]) / dy
v[:, -1] = Uinf * (phi[:, -1] - phi[:, -2]) / dy

va = np.sqrt(u ** 2 + v ** 2)  # 流速の大きさ

# Prepare data for CSV
results = []
for k in range(kmax):
    for j in range(jmax):
        results.append([X[k, j], Y[k, j], phi[j, k], u[j, k], v[j, k], va[j, k]])

# Save to CSV
columns = ["x", "y", "phi", "u", "v", "v_a"]
df = pd.DataFrame(results, columns=columns)
df.to_csv("C:\\\\labo\\\\GPT\\\\flow_data.csv", index=False)
print("Results saved to C:\\\\labo\\\\GPT\\\\flow_data.csv")

# Visualization
fig, ax1 = plt.subplots(figsize=(9, 3), dpi=100)
plt.rcParams["font.size"] = 25
cnt = plt.contourf(X, Y, va.transpose(1, 0), cmap='viridis', levels=100)

sty = np.arange(0.02, ye, 0.05)
stx = np.full(len(sty), -1.0)
xf = np.linspace(0, 1, 100)
f = 0.2 * xf * (1 - xf)
plt.fill_between(xf, f, color='gray')
plt.plot(xf, f, color='gray')

startpoints = np.array([stx, sty]).transpose(1, 0)
plt.streamplot(X, Y, u.transpose(1, 0), v.transpose(1, 0),
               color='white', start_points=startpoints, linewidth=0.5, arrowstyle='-')

plt.xlabel('x', fontsize=25)
plt.ylabel('y', fontsize=25)
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)
plt.xticks([xs, 0, 1, xe])
plt.subplots_adjust(left=0.17)
plt.colorbar(orientation='horizontal')
plt.show()

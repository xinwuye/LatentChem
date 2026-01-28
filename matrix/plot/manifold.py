import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- Ackley function ----------
def rastrigin(x, y):
    return (
        20 + x**2 + y**2
        - 10 * (np.cos(2*np.pi*x) + np.cos(2*np.pi*y))
    )

# Z = rastrigin(X, y)


# ---------- Grid ----------
n = 400
x = np.linspace(-4, 4, n)
y = np.linspace(-4, 4, n)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

# ---------- Plot ----------
fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    X, Y, Z,
    rstride=2, cstride=2,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.95
)

ax.view_init(elev=45, azim=135)
ax.set_axis_off()
plt.tight_layout()
plt.show()

plt.savefig("manifold.png", dpi=300)

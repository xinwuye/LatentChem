import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======================
# 1. 示例数据（可替换）
# ======================
np.random.seed(42)
N = 60000

x = np.random.randn(N)
y = 0.6 * x + np.random.randn(N) * 0.5

# ======================
# 2. 球形化（关键：让分布变圆）
# ======================
X = np.stack([x, y], axis=1)
X = (X - X.mean(axis=0)) / X.std(axis=0)

# whitening（去相关）
cov = np.cov(X.T)
eigval, eigvec = np.linalg.eigh(cov)
X = X @ eigvec @ np.diag(1.0 / np.sqrt(eigval))

x, y = X[:, 0], X[:, 1]

# ======================
# 3. 轻微 subsample（避免中心糊死）
# ======================
keep = np.random.choice(len(x), size=int(0.85 * len(x)), replace=False)
x, y = x[keep], y[keep]

# ======================
# 4. 画布
# ======================
fig, ax = plt.subplots(figsize=(6, 6))

# ======================
# 5. 底层：细 + 深灰
# ======================
ax.scatter(
    x,
    y,
    s=0.8,
    alpha=0.08,          # 比之前明显更深
    c="#555555",         # 深灰
    linewidths=0
)

# ======================
# 6. 上层：稍粗 + 更深
# ======================
idx = np.random.choice(len(x), size=3500, replace=False)
ax.scatter(
    x[idx],
    y[idx],
    s=10.0,
    alpha=0.35,
    c="#333333",         # 更深灰
    linewidths=0
)

# ======================
# 7. 强制圆形视觉
# ======================
ax.set_aspect("equal", adjustable="box")

# ======================
# 8. 去掉所有装饰
# ======================
ax.set_axis_off()
for spine in ax.spines.values():
    spine.set_visible(False)

plt.margins(0)
plt.tight_layout(pad=0)

# ======================
# 9. 保存透明背景
# ======================
plt.savefig(
    "dense_scatter_round_dark.png",
    dpi=600,
    transparent=True,
    bbox_inches="tight",
    pad_inches=0
)
plt.close()

print("Saved: dense_scatter_round_dark.png")

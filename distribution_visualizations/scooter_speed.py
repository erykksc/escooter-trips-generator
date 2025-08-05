import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, triang, skewnorm, uniform

# Speed range
speed_vals = np.linspace(0, 20, 1000)

# ---------- 1. Beta Distribution ----------
# More sharply peaked Beta
alpha = 8
beta_param = 1

# Rescale Beta from [0,1] to [0,20]
x_beta = speed_vals / 20
pdf_beta = beta.pdf(x_beta, alpha, beta_param) / 20  # divide by scale for density

# ---------- 2. Triangular Distribution ----------
# Triangular distribution with mode near 20 km/h
c = (19.5 - 0) / 20
pdf_tri = triang.pdf(speed_vals, c=c, loc=0, scale=20)

# ---------- 3. Skew-Normal Distribution ----------
# Skew-normal centered near 20 and skewed left
a = -8
loc = 20
scale = 5

x_skew = np.linspace(-10, 30, 2000)
pdf_skew_full = skewnorm.pdf(x_skew, a=a, loc=loc, scale=scale)

# Truncate to 0–20 km/h
mask = (x_skew >= 0) & (x_skew <= 20)
speed_skew = x_skew[mask]
pdf_skew = pdf_skew_full[mask]
# Normalize the truncated density
pdf_skew /= np.trapz(pdf_skew, speed_skew)

# ---------- 4. Uniform Distribution ----------
pdf_uniform = uniform.pdf(speed_vals, loc=0, scale=20)

# ---------- Plot ----------
plt.figure(figsize=(10, 6))

plt.plot(
    speed_vals, pdf_beta, label=f"Beta({alpha},{beta_param})", color="blue", linewidth=2
)
plt.plot(
    speed_vals,
    pdf_tri,
    label="Triangular(0,19.5,20)",
    color="green",
    linestyle="--",
    linewidth=2,
)
plt.plot(
    speed_skew,
    pdf_skew,
    label="Skew-Normal (truncated)",
    color="red",
    linestyle="-.",
    linewidth=2,
)
plt.plot(
    speed_vals,
    pdf_uniform,
    label="Uniform(0,20)",
    color="gray",
    linestyle=":",
    linewidth=2,
)

plt.xlabel("Speed (km/h)")
plt.ylabel("Density")
plt.title("Left-Skewed Distributions for Escooter Speed (Sharper Beta Peak)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

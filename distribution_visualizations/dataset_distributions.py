#!/usr/bin/env python3
"""
Visualization of the actual distributions used in the dataset generator.
This script plots all the probability distributions that are implemented in dataset-generator.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, lognorm, uniform
from datetime import datetime
from zoneinfo import ZoneInfo

# Set random seed for reproducibility (same as in dataset-generator.py)
np.random.seed(14)

fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Speed Distribution (Beta Distribution)
ax1 = axes[0, 0]
alpha = 8
beta_param = 1
speed_vals = np.linspace(0, 20, 1000)
x_beta = speed_vals / 20  # normalize to [0,1] for beta distribution
pdf_beta = beta.pdf(x_beta, alpha, beta_param) / 20  # scale back for density

ax1.plot(
    speed_vals, pdf_beta, "b-", linewidth=2, label=f"Beta(α={alpha}, β={beta_param})"
)
ax1.fill_between(speed_vals, pdf_beta, alpha=0.3, color="blue")
ax1.set_xlabel("Speed (km/h)")
ax1.set_ylabel("Probability Density")
ax1.set_title("E-Scooter Speed Distribution\n(Beta Distribution scaled to 0-20 km/h)")
ax1.grid(True, alpha=0.3)
ax1.legend()

# Add statistics
mean_speed = alpha / (alpha + beta_param) * 20
ax1.axvline(
    mean_speed,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Mean: {mean_speed:.1f} km/h",
)
ax1.legend()

# Route Length Distribution (Log-normal Distribution)
ax2 = axes[0, 1]
shape = 0.5  # σ of the underlying normal
scale = 2100  # median of the distribution
distance_vals = np.linspace(1, 15000, 1000)
pdf_lognorm = lognorm.pdf(distance_vals, shape, loc=0, scale=scale)

ax2.plot(
    distance_vals,
    pdf_lognorm,
    "g-",
    linewidth=2,
    label=f"Log-normal(σ={shape}, scale={scale}m)",
)
ax2.fill_between(distance_vals, pdf_lognorm, alpha=0.3, color="green")
ax2.set_xlabel("Route Length (meters)")
ax2.set_ylabel("Probability Density")
ax2.set_title("Route Length Distribution\n(Log-normal Distribution)")
ax2.grid(True, alpha=0.3)

# Add statistics
median_length = scale
mean_length = scale * np.exp(shape**2 / 2)
mode_length = scale * np.exp(-(shape**2))
ax2.axvline(
    median_length,
    color="red",
    linestyle="--",
    alpha=0.7,
    label=f"Median: {median_length}m",
)
ax2.axvline(
    mean_length,
    color="orange",
    linestyle="--",
    alpha=0.7,
    label=f"Mean: {mean_length:.0f}m",
)
ax2.axvline(
    mode_length,
    color="purple",
    linestyle="--",
    alpha=0.7,
    label=f"Mode: {mode_length:.0f}m",
)
ax2.legend()

# Start Point Selection (Uniform)
ax3 = axes[1, 0]
# This is conceptual since it's uniform over graph nodes
node_indices = np.arange(0, 100)  # example with 100 nodes
uniform_prob = np.ones(len(node_indices)) / len(node_indices)

ax3.bar(node_indices, uniform_prob, width=0.8, alpha=0.7, color="orange")
ax3.set_xlabel("Graph Node Index (example)")
ax3.set_ylabel("Selection Probability")
ax3.set_title("Start Point Selection\n(Uniform Distribution over Graph Nodes)")
ax3.grid(True, alpha=0.3)
ax3.text(
    0.5,
    0.8,
    "Each node has equal\nselection probability",
    transform=ax3.transAxes,
    ha="center",
    va="center",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Random Bearing Distribution (Uniform)
ax4 = axes[1, 1]
bearing_vals = np.linspace(0, 360, 1000)
pdf_bearing = uniform.pdf(bearing_vals, loc=0, scale=360)

ax4.plot(bearing_vals, pdf_bearing, "r-", linewidth=2, label="Uniform(0°, 360°)")
ax4.fill_between(bearing_vals, pdf_bearing, alpha=0.3, color="red")
ax4.set_xlabel("Bearing (degrees)")
ax4.set_ylabel("Probability Density")
ax4.set_title("Random Bearing Distribution\n(Uniform Distribution)")
ax4.grid(True, alpha=0.3)
ax4.legend()

# Start Time Distribution (Uniform)
ax5 = axes[2, 0]
# Time range from dataset-generator.py
berlin_tz = ZoneInfo("Europe/Berlin")
start_ts = datetime(2020, 1, 1, tzinfo=berlin_tz).timestamp()
end_ts = datetime(2025, 12, 31, tzinfo=berlin_tz).timestamp()

# Convert to years
start_year = 2020
end_year = 2025
time_vals = np.linspace(start_year, end_year, 1000)
pdf_time = uniform.pdf(time_vals, loc=start_year, scale=(end_year - start_year))

ax5.plot(
    time_vals, pdf_time, "m-", linewidth=2, label=f"Uniform({start_year}, {end_year})"
)
ax5.fill_between(time_vals, pdf_time, alpha=0.3, color="magenta")
ax5.set_xlabel("Year")
ax5.set_ylabel("Probability Density")
ax5.set_title("Start Time Distribution\n(Uniform Distribution 2020-2025)")
ax5.grid(True, alpha=0.3)
ax5.legend()

axes[2, 1].set_visible(False)

plt.tight_layout()

pdf_file = "./dataset_distributions.pdf"
plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
print(f"Saved as PDF: {pdf_file}")
plt.show()

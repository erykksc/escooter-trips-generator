#!/usr/bin/env python3
"""
Visualization of the actual distributions used in the dataset generator.
This script plots all the probability distributions that are implemented in dataset-generator.py
Each distribution is saved as its own PDF file.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, lognorm, uniform
from datetime import datetime
from zoneinfo import ZoneInfo

# Set random seed for reproducibility (same as in dataset-generator.py)
np.random.seed(14)

# Set default font size for all plots
plt.rcParams.update({"font.size": 17})


def create_speed_distribution(show_labels=True):
    """Speed Distribution (Beta Distribution)"""
    plt.figure(figsize=(8, 6))
    alpha = 8
    beta_param = 1
    speed_vals = np.linspace(0, 20, 1000)
    x_beta = speed_vals / 20  # normalize to [0,1] for beta distribution
    pdf_beta = beta.pdf(x_beta, alpha, beta_param) / 20  # scale back for density

    plt.plot(
        speed_vals,
        pdf_beta,
        "b-",
        linewidth=2,
        label=f"Beta(α={alpha}, β={beta_param})",
    )
    plt.fill_between(speed_vals, pdf_beta, alpha=0.3, color="blue")
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Probability Density")
    if show_labels:
        plt.title(
            "E-Scooter Speed Distribution\n(Beta Distribution scaled to 0-20 km/h)"
        )
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_speed = alpha / (alpha + beta_param) * 20
    plt.axvline(
        mean_speed,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_speed:.1f} km/h",
    )
    plt.legend()

    plt.tight_layout()
    pdf_file = (
        "./speed-distribution-no-labels.pdf"
        if not show_labels
        else "./speed-distribution.pdf"
    )
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"Saved speed distribution as PDF: {pdf_file}")
    plt.show()


def create_route_length_distribution(show_labels=True):
    """Route Length Distribution (Log-normal Distribution)"""
    plt.figure(figsize=(8, 6))
    shape = 0.5  # σ of the underlying normal
    scale = 2100  # median of the distribution
    distance_vals = np.linspace(1, 15000, 1000)
    pdf_lognorm = lognorm.pdf(distance_vals, shape, loc=0, scale=scale)

    plt.plot(
        distance_vals,
        pdf_lognorm,
        "g-",
        linewidth=2,
        label=f"Log-normal(σ={shape}, scale={scale}m)",
    )
    plt.fill_between(distance_vals, pdf_lognorm, alpha=0.3, color="green")
    plt.xlabel("Route Length (meters)")
    plt.ylabel("Probability Density")
    if show_labels:
        plt.title("Route Length Distribution\n(Log-normal Distribution)")
    plt.grid(True, alpha=0.3)

    # Add statistics
    median_length = scale
    mean_length = scale * np.exp(shape**2 / 2)
    mode_length = scale * np.exp(-(shape**2))
    plt.axvline(
        median_length,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Median: {median_length}m",
    )
    plt.axvline(
        mean_length,
        color="orange",
        linestyle="--",
        alpha=0.7,
        label=f"Mean: {mean_length:.0f}m",
    )
    plt.axvline(
        mode_length,
        color="purple",
        linestyle="--",
        alpha=0.7,
        label=f"Mode: {mode_length:.0f}m",
    )
    plt.legend()

    plt.tight_layout()
    pdf_file = (
        "./route-length-distribution-no-labels.pdf"
        if not show_labels
        else "./route-length-distribution.pdf"
    )
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"Saved route length distribution as PDF: {pdf_file}")
    plt.show()


def create_start_point_distribution(show_labels=True):
    """Start Point Selection (Uniform)"""
    plt.figure(figsize=(8, 6))
    # This is conceptual since it's uniform over graph nodes
    node_indices = np.arange(0, 100)  # example with 100 nodes
    uniform_prob = np.ones(len(node_indices)) / len(node_indices)

    plt.bar(node_indices, uniform_prob, width=0.8, alpha=0.7, color="orange")
    plt.xlabel("Graph Node Index (example)")
    plt.ylabel("Selection Probability")
    if show_labels:
        plt.title("Start Point Selection\n(Uniform Distribution over Graph Nodes)")
    plt.grid(True, alpha=0.3)
    plt.text(
        0.5,
        0.8,
        "Each node has equal\nselection probability",
        transform=plt.gca().transAxes,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    pdf_file = (
        "./start-point-distribution-no-labels.pdf"
        if not show_labels
        else "./start-point-distribution.pdf"
    )
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"Saved start point distribution as PDF: {pdf_file}")
    plt.show()


def create_bearing_distribution(show_labels=True):
    """Random Bearing Distribution (Uniform)"""
    plt.figure(figsize=(8, 6))
    bearing_vals = np.linspace(0, 360, 1000)
    pdf_bearing = uniform.pdf(bearing_vals, loc=0, scale=360)

    plt.plot(bearing_vals, pdf_bearing, "r-", linewidth=2, label="Uniform(0°, 360°)")
    plt.fill_between(bearing_vals, pdf_bearing, alpha=0.3, color="red")
    plt.xlabel("Bearing (degrees)")
    plt.ylabel("Probability Density")
    if show_labels:
        plt.title("Random Bearing Distribution\n(Uniform Distribution)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    pdf_file = (
        "./bearing-distribution-no-labels.pdf"
        if not show_labels
        else "./bearing-distribution.pdf"
    )
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"Saved bearing distribution as PDF: {pdf_file}")
    plt.show()


def create_time_distribution(show_labels=True):
    """Start Time Distribution (Uniform)"""
    plt.figure(figsize=(8, 6))
    # Time range from dataset-generator.py
    berlin_tz = ZoneInfo("Europe/Berlin")
    start_ts = datetime(2020, 1, 1, tzinfo=berlin_tz).timestamp()
    end_ts = datetime(2025, 12, 31, tzinfo=berlin_tz).timestamp()

    # Convert to years
    start_year = 2020
    end_year = 2025
    time_vals = np.linspace(start_year, end_year, 1000)
    pdf_time = uniform.pdf(time_vals, loc=start_year, scale=(end_year - start_year))

    plt.plot(
        time_vals,
        pdf_time,
        "m-",
        linewidth=2,
        label=f"Uniform({start_year}, {end_year})",
    )
    plt.fill_between(time_vals, pdf_time, alpha=0.3, color="magenta")
    plt.xlabel("Year")
    plt.ylabel("Probability Density")
    if show_labels:
        plt.title("Start Time Distribution\n(Uniform Distribution 2020-2025)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    pdf_file = (
        "./time-distribution-no-labels.pdf"
        if not show_labels
        else "./time-distribution.pdf"
    )
    plt.savefig(pdf_file, dpi=300, bbox_inches="tight")
    print(f"Saved time distribution as PDF: {pdf_file}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate distribution plots for the e-scooter dataset generator"
    )
    parser.add_argument(
        "--no-labels", action="store_true", help="Generate plots without titles/labels"
    )
    args = parser.parse_args()

    show_labels = not args.no_labels
    label_status = "without labels" if args.no_labels else "with labels"

    print(f"Creating individual distribution plots {label_status}...")
    create_speed_distribution(show_labels)
    create_route_length_distribution(show_labels)
    create_start_point_distribution(show_labels)
    create_bearing_distribution(show_labels)
    create_time_distribution(show_labels)
    print(
        f"All distribution plots have been created and saved as separate PDF files {label_status}."
    )

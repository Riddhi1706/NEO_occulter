#!/usr/bin/env python3
"""
analyze_neo_catalogs.py

Initial population-level analysis for cleaned JPL NEO catalogs.

Expected folder structure:
project/
├── analyze_neo_catalogs.py
└── cleaned data/
    ├── atira_cleaned.csv
    ├── aten_cleaned.csv
    ├── apollo_cleaned.csv
    └── amor_cleaned.csv

Outputs:
analysis results/
├── class_summary.csv
├── pha_by_class.png
├── moid_distribution.png
├── orbital_geometry.png
└── physical_data_completeness.png

This is deliberately a catalog-level analysis only.
Solar elongation requires propagating each asteroid and the observer in time,
which should be handled in the next script.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = Path("cleaned data")
OUTPUT_DIR = Path("analysis results")

FILES = {
    "Atira": DATA_DIR / "atira_cleaned.csv",
    "Aten": DATA_DIR / "aten_cleaned.csv",
    "Apollo": DATA_DIR / "apollo_cleaned.csv",
    "Amor": DATA_DIR / "amor_cleaned.csv",
}


def load_catalogs():
    """Load all four cleaned NEO catalogs and append a class label."""
    frames = []

    for neo_class, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find {path}. "
                "Check that the cleaned CSV filenames match FILES."
            )

        df = pd.read_csv(path)
        df["neo_class"] = neo_class
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def normalize_pha(series):
    """
    Make PHA robust to CSV representations such as:
    True/False, Y/N, 1/0.
    """
    text = series.astype("string").str.strip().str.lower()

    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(["true", "y", "yes", "1"])] = True
    result[text.isin(["false", "n", "no", "0"])] = False

    return result


def build_summary(df):
    """Generate one summary row per NEO class."""
    rows = []

    for neo_class, group in df.groupby("neo_class", sort=False):
        pha = group["PHA_bool"]

        rows.append(
            {
                "class": neo_class,
                "total_objects": len(group),
                "PHA_count": int((pha == True).sum()),
                "PHA_fraction_percent": 100.0 * (pha == True).mean(),
                "median_MOID_au": group["moid_au"].median(),
                "min_MOID_au": group["moid_au"].min(),
                "median_q_au": group["q_au"].median(),
                "median_Q_au": group["Q_au"].median(),
                "median_period_years": group["period_years"].median(),
                "median_H": group["H"].median(),
                "diameter_available_percent": 100.0
                * group["diameter_km"].notna().mean(),
                "albedo_available_percent": 100.0
                * group["albedo"].notna().mean(),
            }
        )

    return pd.DataFrame(rows)


def plot_pha_counts(summary):
    """Total objects and PHA objects by class."""
    x = np.arange(len(summary))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        summary["total_objects"],
        width,
        label="All objects",
    )
    ax.bar(
        x + width / 2,
        summary["PHA_count"],
        width,
        label="PHA",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary["class"])
    ax.set_ylabel("Number of asteroids")
    ax.set_title("NEO population and potentially hazardous asteroids")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "pha_by_class.png", dpi=200)
    plt.close(fig)


def plot_moid_distribution(df):
    """Compare Earth MOID distributions by NEO class."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins = np.linspace(0, 0.3, 61)

    for neo_class in FILES:
        values = df.loc[df["neo_class"] == neo_class, "moid_au"].dropna()

        # Keep the near-Earth portion visually useful.
        values = values[values <= 0.3]

        if len(values):
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.5,
                label=neo_class,
            )

    ax.axvline(
        0.05,
        linestyle="--",
        linewidth=1.2,
        label="0.05 AU PHA proximity threshold",
    )

    ax.set_xlabel("Earth MOID (AU)")
    ax.set_ylabel("Number of asteroids")
    ax.set_title("Earth minimum-orbit-intersection distance")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "moid_distribution.png", dpi=200)
    plt.close(fig)


def plot_orbital_geometry(df):
    """
    Show where the four orbit classes sit in perihelion-aphelion space.

    This is useful before time propagation because it exposes the basic
    geometric differences in solar proximity between the populations.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for neo_class in FILES:
        group = df[df["neo_class"] == neo_class]

        ax.scatter(
            group["q_au"],
            group["Q_au"],
            s=12,
            alpha=0.45,
            label=neo_class,
        )

    ax.set_xlabel("Perihelion distance q (AU)")
    ax.set_ylabel("Aphelion distance Q (AU)")
    ax.set_title("NEO orbital geometry")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "orbital_geometry.png", dpi=200)
    plt.close(fig)


def plot_physical_completeness(df):
    """
    Check whether JPL supplies enough physical data for direct IR modeling.

    Diameter is especially important for thermal-IR flux.
    """
    rows = []

    for neo_class, group in df.groupby("neo_class", sort=False):
        rows.append(
            {
                "class": neo_class,
                "diameter": 100 * group["diameter_km"].notna().mean(),
                "albedo": 100 * group["albedo"].notna().mean(),
                "H": 100 * group["H"].notna().mean(),
            }
        )

    completeness = pd.DataFrame(rows)

    x = np.arange(len(completeness))
    width = 0.24

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width,
        completeness["diameter"],
        width,
        label="Diameter",
    )
    ax.bar(
        x,
        completeness["albedo"],
        width,
        label="Albedo",
    )
    ax.bar(
        x + width,
        completeness["H"],
        width,
        label="H",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(completeness["class"])
    ax.set_ylabel("Catalog completeness (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Physical-property coverage for IR modeling")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "physical_data_completeness.png",
        dpi=200,
    )
    plt.close(fig)


def print_summary(summary):
    print("\nNEO CATALOG SUMMARY")
    print("=" * 80)

    display_cols = [
        "class",
        "total_objects",
        "PHA_count",
        "PHA_fraction_percent",
        "median_MOID_au",
        "median_q_au",
        "median_Q_au",
        "median_period_years",
        "diameter_available_percent",
        "albedo_available_percent",
    ]

    print(
        summary[display_cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_catalogs()

    # Normalize fields that may have been read back from CSV differently.
    df["PHA_bool"] = normalize_pha(df["PHA"])

    numeric_fields = [
        "a_au",
        "e",
        "i_deg",
        "Omega_deg",
        "omega_deg",
        "q_au",
        "Q_au",
        "period_years",
        "period_days",
        "epoch_jd",
        "M_deg",
        "moid_au",
        "H",
        "diameter_km",
        "diameter_sigma_km",
        "albedo",
        "data_arc_days",
        "condition_code",
        "n_obs_used",
        "n_delay_obs_used",
        "n_doppler_obs_used",
    ]

    for column in numeric_fields:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    summary = build_summary(df)
    summary.to_csv(OUTPUT_DIR / "class_summary.csv", index=False)

    print_summary(summary)

    plot_pha_counts(summary)
    plot_moid_distribution(df)
    plot_orbital_geometry(df)
    plot_physical_completeness(df)

    print("\nSaved analysis to:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()

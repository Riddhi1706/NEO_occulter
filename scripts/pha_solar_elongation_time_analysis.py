#!/usr/bin/env python3
"""
pha_solar_elongation_time_analysis.py

Companion analysis for the solar-elongation project.

Purpose
-------
1. Restrict the detailed time-history analysis to JPL-designated PHAs.
2. Propagate each PHA over the full Earth ephemeris once.
3. Compare minimum solar elongation over shorter mission windows:
       1 year, 3 years, 5 years, 10 years
4. Save an S-O-T (solar elongation) versus time plot for every PHA.
5. Save compact CSV summaries for later mission-design interpretation.

This script imports the orbital-propagation physics from:
    solar_elongation_analysis.py

Expected folder structure
-------------------------
project/
├── solar_elongation_analysis.py
├── pha_solar_elongation_time_analysis.py
└── cleaned data/
    ├── earth_ephemeris.csv
    ├── atira_cleaned.csv
    ├── aten_cleaned.csv
    ├── apollo_cleaned.csv
    └── amor_cleaned.csv

Outputs
-------
solar elongation results/
├── pha_short_window_summary.csv
├── pha_short_window_threshold_summary.csv
├── pha_min_elongation_by_window_Atira.png
├── pha_min_elongation_by_window_Aten.png
├── pha_min_elongation_by_window_Apollo.png
├── pha_min_elongation_by_window_Amor.png
├── pha_threshold_fraction_vs_window.png
└── PHA time histories/
    ├── Atira/
    ├── Aten/
    ├── Apollo/
    └── Amor/

Notes
-----
The asteroid propagation remains a two-body Keplerian approximation.
The Earth trajectory is taken directly from JPL Horizons.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solar_elongation_analysis import (
    compute_solar_elongation,
    load_earth_ephemeris,
    load_neo_catalog,
    propagate_chunk,
    valid_orbit_mask,
)


NEO_FILES = {
    "Atira": "atira_cleaned.csv",
    "Aten": "aten_cleaned.csv",
    "Apollo": "apollo_cleaned.csv",
    "Amor": "amor_cleaned.csv",
}

WINDOWS_YEARS = [1, 3, 5, 10]
THRESHOLDS_DEG = [5.0, 10.0, 20.0, 30.0]
HIST_BINS_DEG = np.arange(0.0, 65.0, 5.0)


def horizons_dates_to_datetime(series: pd.Series) -> pd.DatetimeIndex:
    """
    Convert strings like:
        A.D. 2026-Sep-03 00:00:00.0000
    to pandas datetime values for plotting.
    """
    cleaned = (
        series.astype(str)
        .str.replace("A.D. ", "", regex=False)
        .str.strip()
    )

    return pd.to_datetime(
        cleaned,
        format="%Y-%b-%d %H:%M:%S.%f",
        errors="coerce",
    )


def safe_filename(text: str) -> str:
    """Make asteroid names safe for use as filenames."""
    text = str(text).strip()
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text[:120]


def make_window_masks(jd: np.ndarray) -> dict[int, np.ndarray]:
    """
    Build masks for the first 1, 3, 5, and 10 years.

    We use Julian-date duration rather than calendar-year slicing because
    the orbital propagation itself is performed in continuous TDB days.
    """
    start_jd = jd[0]

    masks = {}

    for years in WINDOWS_YEARS:
        end_jd = start_jd + years * 365.25
        masks[years] = jd <= end_jd

    return masks


def build_short_window_rows(
    chunk: pd.DataFrame,
    elong_deg: np.ndarray,
    jd: np.ndarray,
    dates: np.ndarray,
    window_masks: dict[int, np.ndarray],
    neo_class: str,
) -> list[dict]:
    """
    For every PHA in the chunk, compute the minimum elongation separately
    over 1, 3, 5, and 10 year windows.
    """
    rows = []

    for object_index, (_, asteroid) in enumerate(chunk.iterrows()):
        for years, mask in window_masks.items():
            e_series = elong_deg[object_index, mask]
            jd_series = jd[mask]
            date_series = dates[mask]

            min_local_idx = int(np.argmin(e_series))

            row = {
                "name": asteroid["name"],
                "spkid": asteroid["spkid"],
                "neo_class": neo_class,
                "window_years": years,
                "n_daily_samples": int(mask.sum()),
                "min_elong_deg": float(e_series[min_local_idx]),
                "min_elong_jd": float(jd_series[min_local_idx]),
                "min_elong_date_tdb": str(date_series[min_local_idx]),
                "mean_elong_deg": float(np.mean(e_series)),
                "median_elong_deg": float(np.median(e_series)),
                "fraction_below_5deg": float(np.mean(e_series < 5.0)),
                "fraction_below_10deg": float(np.mean(e_series < 10.0)),
                "fraction_below_20deg": float(np.mean(e_series < 20.0)),
                "fraction_below_30deg": float(np.mean(e_series < 30.0)),
            }

            # Preserve useful catalog context when available.
            for col in [
                "moid_au",
                "H",
                "q_au",
                "Q_au",
                "period_years",
                "diameter_km",
                "albedo",
            ]:
                if col in asteroid.index:
                    row[col] = asteroid[col]

            rows.append(row)

    return rows


def plot_single_pha_time_history(
    asteroid: pd.Series,
    time_dates: pd.DatetimeIndex,
    elong_deg: np.ndarray,
    output_dir: Path,
    neo_class: str,
):
    """
    Save one full-span S-O-T versus time plot for a single PHA.
    """
    fig, ax = plt.subplots(figsize=(11, 5.5))

    ax.plot(time_dates, elong_deg, linewidth=1.0)

    # Reference solar-elongation regions relevant to the mission trade.
    for threshold in THRESHOLDS_DEG:
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )

    min_idx = int(np.argmin(elong_deg))
    min_angle = float(elong_deg[min_idx])
    min_date = time_dates[min_idx]

    ax.scatter(
        [min_date],
        [min_angle],
        s=35,
        zorder=4,
        label=f"Minimum = {min_angle:.2f}°",
    )

    name = str(asteroid["name"]).strip()
    spkid = str(asteroid["spkid"]).strip()

    ax.set_title(
        f"{name} — Earth-centered solar elongation over time\n"
        f"{neo_class} PHA | SPK-ID {spkid}"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Sun–Observer–Target elongation (deg)")
    ax.set_ylim(0, 180)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")

    fig.tight_layout()

    class_dir = output_dir / "PHA time histories" / neo_class
    class_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{safe_filename(name)}_"
        f"{safe_filename(spkid)}_SOT.png"
    )

    fig.savefig(class_dir / filename, dpi=160)
    plt.close(fig)


def plot_window_histograms(
    short_summary: pd.DataFrame,
    output_dir: Path,
):
    """
    For each NEO class, overlay the minimum-elongation distributions from
    the 1, 3, 5, and 10 year windows.

    This directly reveals how the 'minimum elongation' statistic migrates
    toward smaller values as the observation window becomes longer.
    """
    for neo_class in NEO_FILES:
        group = short_summary[
            short_summary["neo_class"] == neo_class
        ]

        if group.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 5.5))

        for years in WINDOWS_YEARS:
            values = group.loc[
                group["window_years"] == years,
                "min_elong_deg",
            ].dropna()

            ax.hist(
                values,
                bins=HIST_BINS_DEG,
                histtype="step",
                linewidth=1.7,
                label=f"{years} year",
            )

        ax.set_xlim(0, 60)
        ax.set_xlabel("Minimum solar elongation reached (deg)")
        ax.set_ylabel("Number of PHA asteroids")
        ax.set_title(
            f"{neo_class} PHAs — effect of propagation window "
            "on minimum elongation"
        )
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"pha_min_elongation_by_window_{neo_class}.png",
            dpi=200,
        )
        plt.close(fig)


def build_threshold_summary(
    short_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize what fraction of unique PHAs enter each low-elongation region
    within each propagation window.
    """
    rows = []

    for neo_class in NEO_FILES:
        for years in WINDOWS_YEARS:
            group = short_summary[
                (short_summary["neo_class"] == neo_class)
                & (short_summary["window_years"] == years)
            ]

            for threshold in THRESHOLDS_DEG:
                entering = group["min_elong_deg"] <= threshold

                rows.append({
                    "neo_class": neo_class,
                    "window_years": years,
                    "threshold_deg": threshold,
                    "PHA_count": len(group),
                    "PHA_entering_count": int(entering.sum()),
                    "PHA_entering_fraction_percent": (
                        100.0 * entering.mean()
                        if len(group) else np.nan
                    ),
                })

    return pd.DataFrame(rows)


def plot_threshold_vs_window(
    threshold_summary: pd.DataFrame,
    output_dir: Path,
):
    """
    Create a compact comparison plot showing how the percentage of PHAs
    entering low-elongation regions changes as the mission window increases.

    One figure is produced for each threshold to avoid mixing several
    different mission criteria on one axis.
    """
    for threshold in THRESHOLDS_DEG:
        subset = threshold_summary[
            threshold_summary["threshold_deg"] == threshold
        ]

        fig, ax = plt.subplots(figsize=(8.5, 5.5))

        for neo_class in NEO_FILES:
            group = subset[
                subset["neo_class"] == neo_class
            ]

            ax.plot(
                group["window_years"],
                group["PHA_entering_fraction_percent"],
                marker="o",
                label=neo_class,
            )

        ax.set_xticks(WINDOWS_YEARS)
        ax.set_xlabel("Propagation / survey window (years)")
        ax.set_ylabel("PHAs entering region (%)")
        ax.set_ylim(0, 100)
        ax.set_title(
            f"PHAs reaching solar elongation ≤ {threshold:.0f}°"
        )
        ax.grid(alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            output_dir
            / f"pha_fraction_reaching_{int(threshold)}deg_vs_window.png",
            dpi=200,
        )
        plt.close(fig)


def analyze_pha_class(
    catalog: pd.DataFrame,
    earth: pd.DataFrame,
    neo_class: str,
    output_dir: Path,
    chunk_size: int,
    make_time_plots: bool,
):
    """
    Propagate the valid PHAs in one class over the full Horizons interval.
    """
    pha = catalog[
        catalog["PHA_bool"].fillna(False)
    ].copy()

    valid = valid_orbit_mask(pha)

    skipped = pha.loc[
        ~valid,
        ["name", "spkid"],
    ].copy()

    skipped["neo_class"] = neo_class
    skipped["reason"] = "missing or invalid propagation elements"

    pha = pha.loc[valid].reset_index(drop=True)

    jd = earth["jd_tdb"].to_numpy(float)
    date_strings = earth["date_tdb"].astype(str).to_numpy()
    plot_dates = horizons_dates_to_datetime(earth["date_tdb"])

    r_earth = earth[
        ["x_au", "y_au", "z_au"]
    ].to_numpy(float)

    window_masks = make_window_masks(jd)

    rows = []

    for start in range(0, len(pha), chunk_size):
        stop = min(start + chunk_size, len(pha))
        chunk = pha.iloc[start:stop].copy()

        r_ast, _ = propagate_chunk(chunk, jd)

        elong_deg, _ = compute_solar_elongation(
            r_ast,
            r_earth,
        )

        rows.extend(
            build_short_window_rows(
                chunk,
                elong_deg,
                jd,
                date_strings,
                window_masks,
                neo_class,
            )
        )

        if make_time_plots:
            for local_index, (_, asteroid) in enumerate(
                chunk.iterrows()
            ):
                plot_single_pha_time_history(
                    asteroid,
                    plot_dates,
                    elong_deg[local_index],
                    output_dir,
                    neo_class,
                )

        print(
            f"{neo_class}: propagated "
            f"{stop}/{len(pha)} PHAs",
            end="\r",
        )

    print()

    return pd.DataFrame(rows), skipped


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 1/3/5/10-year solar-elongation behavior "
            "and generate PHA S-O-T time histories."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("cleaned data"),
        help="Directory containing cleaned NEO catalogs.",
    )

    parser.add_argument(
        "--earth",
        type=Path,
        default=Path("cleaned data") / "earth_ephemeris.csv",
        help="Cleaned JPL Horizons Earth ephemeris.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("solar elongation results"),
        help="Output directory.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of PHAs propagated simultaneously.",
    )

    parser.add_argument(
        "--no-time-plots",
        action="store_true",
        help=(
            "Skip the per-asteroid S-O-T PNGs. Useful for quickly "
            "testing the shorter-window statistics first."
        ),
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    earth = load_earth_ephemeris(args.earth)

    print(
        f"Earth ephemeris: {len(earth)} daily samples "
        f"from {earth['date_tdb'].iloc[0]} "
        f"to {earth['date_tdb'].iloc[-1]}"
    )

    all_summaries = []
    skipped_parts = []

    for neo_class, filename in NEO_FILES.items():
        path = args.data_dir / filename

        catalog = load_neo_catalog(
            path,
            neo_class,
        )

        summary, skipped = analyze_pha_class(
            catalog,
            earth,
            neo_class,
            args.output_dir,
            args.chunk_size,
            make_time_plots=not args.no_time_plots,
        )

        all_summaries.append(summary)
        skipped_parts.append(skipped)

    short_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    short_summary.to_csv(
        args.output_dir / "pha_short_window_summary.csv",
        index=False,
    )

    threshold_summary = build_threshold_summary(
        short_summary
    )

    threshold_summary.to_csv(
        args.output_dir
        / "pha_short_window_threshold_summary.csv",
        index=False,
    )

    skipped_df = pd.concat(
        skipped_parts,
        ignore_index=True,
    )

    skipped_df.to_csv(
        args.output_dir / "pha_skipped_objects.csv",
        index=False,
    )

    plot_window_histograms(
        short_summary,
        args.output_dir,
    )

    plot_threshold_vs_window(
        threshold_summary,
        args.output_dir,
    )

    print("\nDone.")
    print(
        f"PHA-window rows saved: {len(short_summary)}"
    )
    print(
        f"Unique propagated PHAs: "
        f"{short_summary[['name','spkid','neo_class']].drop_duplicates().shape[0]}"
    )
    print(
        f"Skipped PHAs: {len(skipped_df)}"
    )

    if args.no_time_plots:
        print("Per-PHA time-history plots were skipped.")
    else:
        print(
            "Per-PHA S-O-T plots saved under: "
            f"{args.output_dir / 'PHA time histories'}"
        )


if __name__ == "__main__":
    main()

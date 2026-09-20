#!/usr/bin/env python3
"""
solar_elongation_analysis.py

Propagate cleaned JPL NEO orbital elements with a two-body heliocentric model
and compute solar elongation as seen from Earth using a JPL Horizons Earth
ephemeris.

Expected files
--------------
cleaned data/
    atira_cleaned.csv
    aten_cleaned.csv
    apollo_cleaned.csv
    amor_cleaned.csv

earth_ephemeris.csv

Expected asteroid columns
-------------------------
name, spkid,
a_au, e, i_deg, Omega_deg, omega_deg,
q_au, Q_au, period_years, period_days,
epoch_jd, M_deg,
moid_au, PHA,
H, diameter_km, diameter_sigma_km, albedo,
...

Expected Earth ephemeris columns
--------------------------------
jd_tdb, date_tdb,
x_au, y_au, z_au,
vx_au_day, vy_au_day, vz_au_day

Physics model
-------------
1. Treat each SBDB element set as a heliocentric osculating Keplerian orbit.
2. Propagate mean anomaly:
       M(t) = M0 + n (t - t0)
3. Solve Kepler's equation:
       M = E - e sin(E)
4. Convert eccentric anomaly E to position in the orbital plane.
5. Rotate the orbital-plane position into the J2000 ecliptic frame.
6. Combine asteroid and Earth heliocentric positions.
7. Compute solar elongation:
       epsilon = angle(Earth->Sun, Earth->asteroid)
8. Reduce the daily time histories into population-level mission metrics.

Important limitation
--------------------
This is a TWO-BODY propagation of the asteroid orbital elements.
It does not include planetary perturbations or close-encounter changes.
The resulting population-level elongation statistics should later be checked
against JPL Horizons for a representative set of asteroids.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AU_KM = 149_597_870.7
SECONDS_PER_DAY = 86_400.0
MU_SUN_KM3_S2 = 1.32712440018e11

# Convert solar gravitational parameter to AU^3/day^2.
MU_SUN = (
    MU_SUN_KM3_S2
    * SECONDS_PER_DAY**2
    / AU_KM**3
)

NEO_FILES = {
    "Atira": "atira_cleaned.csv",
    "Aten": "aten_cleaned.csv",
    "Apollo": "apollo_cleaned.csv",
    "Amor": "amor_cleaned.csv",
}

# Mission-relevant low-elongation thresholds.
THRESHOLDS_DEG = np.array([5.0, 10.0, 20.0, 30.0, 45.0])

# Histogram bins. We retain the full 0-180 degree geometry in CSV output,
# although the plots zoom into the low-elongation region.
ELONGATION_BINS_DEG = np.arange(0.0, 185.0, 5.0)


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

def normalize_pha(series: pd.Series) -> pd.Series:
    """Normalize True/False, Y/N, or 1/0 PHA representations."""
    text = series.astype("string").str.strip().str.lower()

    result = pd.Series(pd.NA, index=series.index, dtype="boolean")
    result[text.isin(["true", "y", "yes", "1"])] = True
    result[text.isin(["false", "n", "no", "0"])] = False

    return result


def load_earth_ephemeris(path: Path) -> pd.DataFrame:
    """Load the heliocentric Earth state vectors."""
    earth = pd.read_csv(path)

    required = ["jd_tdb", "date_tdb", "x_au", "y_au", "z_au"]
    missing = [c for c in required if c not in earth.columns]

    if missing:
        raise ValueError(
            "Earth ephemeris is missing columns: " + ", ".join(missing)
        )

    for col in ["jd_tdb", "x_au", "y_au", "z_au"]:
        earth[col] = pd.to_numeric(earth[col], errors="coerce")

    earth = earth.dropna(
        subset=["jd_tdb", "x_au", "y_au", "z_au"]
    ).copy()

    earth = earth.sort_values("jd_tdb").reset_index(drop=True)

    if len(earth) < 2:
        raise ValueError("Earth ephemeris needs at least two epochs.")

    return earth


def load_neo_catalog(path: Path, neo_class: str) -> pd.DataFrame:
    """Load one cleaned asteroid-class catalog."""
    df = pd.read_csv(path)

    required = [
        "name",
        "spkid",
        "a_au",
        "e",
        "i_deg",
        "Omega_deg",
        "omega_deg",
        "epoch_jd",
        "M_deg",
        "PHA",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name} is missing columns: {', '.join(missing)}"
        )

    numeric = [
        "a_au",
        "e",
        "i_deg",
        "Omega_deg",
        "omega_deg",
        "epoch_jd",
        "M_deg",
    ]

    optional_numeric = [
        "q_au",
        "Q_au",
        "period_years",
        "period_days",
        "moid_au",
        "H",
        "diameter_km",
        "diameter_sigma_km",
        "albedo",
    ]

    for col in numeric + optional_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["PHA_bool"] = normalize_pha(df["PHA"])
    df["neo_class"] = neo_class

    return df


def valid_orbit_mask(df: pd.DataFrame) -> pd.Series:
    """
    Select rows that can be propagated as bound two-body elliptical orbits.
    """
    required = [
        "a_au",
        "e",
        "i_deg",
        "Omega_deg",
        "omega_deg",
        "epoch_jd",
        "M_deg",
    ]

    mask = df[required].notna().all(axis=1)
    mask &= df["a_au"] > 0.0
    mask &= df["e"] >= 0.0
    mask &= df["e"] < 1.0

    return mask


# ---------------------------------------------------------------------------
# Orbital mechanics
# ---------------------------------------------------------------------------

def solve_kepler(
    M: np.ndarray,
    e: np.ndarray,
    tolerance: float = 1e-12,
    max_iterations: int = 15,
) -> np.ndarray:
    """
    Solve M = E - e sin(E) for eccentric anomaly E using Newton-Raphson.

    M and e are broadcast-compatible arrays.
    """
    M = np.mod(M, 2.0 * np.pi)

    # Good initial guesses for low and high eccentricity elliptical orbits.
    E = np.where(e < 0.8, M, np.pi)

    for _ in range(max_iterations):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)

        delta = f / fp
        E = E - delta

        if np.max(np.abs(delta)) < tolerance:
            break

    return E


def propagate_chunk(
    chunk: pd.DataFrame,
    jd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a chunk of asteroids to all requested epochs.

    Returns
    -------
    r_ast : ndarray, shape (N_objects, N_times, 3)
        Heliocentric asteroid positions in AU.
    r_helio : ndarray, shape (N_objects, N_times)
        Heliocentric distance in AU.
    """

    # Shape convention:
    # asteroid quantities -> (N_objects, 1)
    # time quantities     -> (1, N_times)
    a = chunk["a_au"].to_numpy(float)[:, None]
    e = chunk["e"].to_numpy(float)[:, None]

    inc = np.deg2rad(chunk["i_deg"].to_numpy(float))[:, None]
    Omega = np.deg2rad(chunk["Omega_deg"].to_numpy(float))[:, None]
    omega = np.deg2rad(chunk["omega_deg"].to_numpy(float))[:, None]

    epoch = chunk["epoch_jd"].to_numpy(float)[:, None]
    M0 = np.deg2rad(chunk["M_deg"].to_numpy(float))[:, None]

    times = jd[None, :]

    # Mean motion from two-body Keplerian dynamics:
    # n = sqrt(mu/a^3), rad/day
    n = np.sqrt(MU_SUN / a**3)

    # Advance the mean anomaly from each asteroid's own SBDB epoch.
    M = M0 + n * (times - epoch)

    # Solve Kepler's equation for eccentric anomaly E.
    E = solve_kepler(M, e)

    # Position in the orbital/perifocal plane.
    x_p = a * (np.cos(E) - e)
    y_p = a * np.sqrt(1.0 - e**2) * np.sin(E)

    # Rotation coefficients for:
    # R3(Omega) R1(i) R3(omega)
    cO = np.cos(Omega)
    sO = np.sin(Omega)
    ci = np.cos(inc)
    si = np.sin(inc)
    co = np.cos(omega)
    so = np.sin(omega)

    A = cO * co - sO * so * ci
    B = -cO * so - sO * co * ci

    C = sO * co + cO * so * ci
    D = -sO * so + cO * co * ci

    F = so * si
    G = co * si

    x = A * x_p + B * y_p
    y = C * x_p + D * y_p
    z = F * x_p + G * y_p

    r_ast = np.stack((x, y, z), axis=2)
    r_helio = np.sqrt(x**2 + y**2 + z**2)

    return r_ast, r_helio


def compute_solar_elongation(
    r_ast: np.ndarray,
    r_earth: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Earth-centered solar elongation.

    Earth -> Sun:
        s = -r_earth

    Earth -> asteroid:
        rho = r_ast - r_earth

    epsilon = angle(s, rho)

    Returns
    -------
    elong_deg : ndarray, shape (N_objects, N_times)
    delta_au : ndarray, shape (N_objects, N_times)
        Earth-asteroid range.
    """

    # Broadcast Earth position from (N_times, 3) to asteroid/time geometry.
    rho = r_ast - r_earth[None, :, :]
    sun_vector = -r_earth

    delta_au = np.linalg.norm(rho, axis=2)
    sun_distance = np.linalg.norm(sun_vector, axis=1)

    dot = np.sum(
        rho * sun_vector[None, :, :],
        axis=2,
    )

    denominator = delta_au * sun_distance[None, :]

    cos_epsilon = dot / denominator

    # Protect arccos against tiny floating-point excursions outside [-1, 1].
    cos_epsilon = np.clip(cos_epsilon, -1.0, 1.0)

    elong_deg = np.rad2deg(np.arccos(cos_epsilon))

    return elong_deg, delta_au


# ---------------------------------------------------------------------------
# Population analysis
# ---------------------------------------------------------------------------

def analyze_class(
    df: pd.DataFrame,
    earth: pd.DataFrame,
    neo_class: str,
    chunk_size: int,
) -> tuple[pd.DataFrame, dict, list]:
    """
    Propagate one NEO class and reduce the daily elongation histories.

    We intentionally do not save every asteroid/day combination because the
    complete four-class dataset would contain >100 million rows.
    """
    valid = valid_orbit_mask(df)

    skipped = df.loc[~valid, ["name", "spkid"]].copy()
    skipped["neo_class"] = neo_class
    skipped["reason"] = "missing or invalid propagation elements"

    work = df.loc[valid].reset_index(drop=True)

    jd = earth["jd_tdb"].to_numpy(float)
    dates = earth["date_tdb"].astype(str).to_numpy()

    r_earth = earth[["x_au", "y_au", "z_au"]].to_numpy(float)

    # Aggregate solar-elongation histogram over every object-time sample.
    hist_all = np.zeros(len(ELONGATION_BINS_DEG) - 1, dtype=np.int64)
    hist_pha = np.zeros(len(ELONGATION_BINS_DEG) - 1, dtype=np.int64)

    summary_parts = []

    for start in range(0, len(work), chunk_size):
        stop = min(start + chunk_size, len(work))
        chunk = work.iloc[start:stop].copy()

        r_ast, r_helio = propagate_chunk(chunk, jd)
        elong_deg, delta_au = compute_solar_elongation(r_ast, r_earth)

        # ---------------------------------------------------------------
        # Per-object extrema
        # ---------------------------------------------------------------
        min_idx = np.argmin(elong_deg, axis=1)
        max_idx = np.argmax(elong_deg, axis=1)

        row_idx = np.arange(len(chunk))

        min_elong = elong_deg[row_idx, min_idx]
        max_elong = elong_deg[row_idx, max_idx]

        min_jd = jd[min_idx]
        min_date = dates[min_idx]

        delta_at_min = delta_au[row_idx, min_idx]
        r_at_min = r_helio[row_idx, min_idx]

        # ---------------------------------------------------------------
        # Time spent in low-elongation regions
        # ---------------------------------------------------------------
        metrics = {}

        for threshold in THRESHOLDS_DEG:
            label = int(threshold)

            below = elong_deg < threshold

            metrics[f"fraction_below_{label}deg"] = below.mean(axis=1)
            metrics[f"days_below_{label}deg"] = below.sum(axis=1)

        # ---------------------------------------------------------------
        # Save compact asteroid summary
        # ---------------------------------------------------------------
        result = pd.DataFrame({
            "name": chunk["name"].to_numpy(),
            "spkid": chunk["spkid"].to_numpy(),
            "neo_class": neo_class,
            "PHA": chunk["PHA_bool"].to_numpy(),
            "min_elong_deg": min_elong,
            "min_elong_jd": min_jd,
            "min_elong_date_tdb": min_date,
            "max_elong_deg": max_elong,
            "earth_range_at_min_elong_au": delta_at_min,
            "helio_distance_at_min_elong_au": r_at_min,
        })

        # Retain useful catalog quantities for later interpretation.
        for col in [
            "moid_au",
            "H",
            "diameter_km",
            "albedo",
            "q_au",
            "Q_au",
            "period_years",
        ]:
            if col in chunk.columns:
                result[col] = chunk[col].to_numpy()

        for key, values in metrics.items():
            result[key] = values

        summary_parts.append(result)

        # ---------------------------------------------------------------
        # Population histogram accumulation
        # ---------------------------------------------------------------
        hist_all += np.histogram(
            elong_deg.ravel(),
            bins=ELONGATION_BINS_DEG,
        )[0]

        pha_mask = chunk["PHA_bool"].fillna(False).to_numpy(dtype=bool)

        if np.any(pha_mask):
            hist_pha += np.histogram(
                elong_deg[pha_mask].ravel(),
                bins=ELONGATION_BINS_DEG,
            )[0]

        print(
            f"{neo_class}: propagated {stop}/{len(work)} objects",
            end="\r",
        )

    print()

    if summary_parts:
        summary = pd.concat(summary_parts, ignore_index=True)
    else:
        summary = pd.DataFrame()

    histograms = {
        "all": hist_all,
        "pha": hist_pha,
        "n_valid": len(work),
        "n_pha": int(work["PHA_bool"].fillna(False).sum()),
        "n_times": len(jd),
    }

    return summary, histograms, [skipped]


def make_histogram_table(hist_by_class: dict) -> pd.DataFrame:
    """Convert histogram counts into a tidy CSV table."""
    rows = []

    left = ELONGATION_BINS_DEG[:-1]
    right = ELONGATION_BINS_DEG[1:]
    centers = 0.5 * (left + right)

    for neo_class, h in hist_by_class.items():
        all_total = h["all"].sum()
        pha_total = h["pha"].sum()

        for i in range(len(left)):
            rows.append({
                "neo_class": neo_class,
                "bin_left_deg": left[i],
                "bin_right_deg": right[i],
                "bin_center_deg": centers[i],
                "all_sample_count": int(h["all"][i]),
                "all_sample_fraction_percent": (
                    100.0 * h["all"][i] / all_total
                    if all_total else np.nan
                ),
                "PHA_sample_count": int(h["pha"][i]),
                "PHA_sample_fraction_percent": (
                    100.0 * h["pha"][i] / pha_total
                    if pha_total else np.nan
                ),
            })

    return pd.DataFrame(rows)


def make_threshold_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Count unique objects that ENTER within each low-elongation threshold.

    Example:
        min_elong_deg <= 10
    means that the object enters the <10 degree region at least once during
    the simulation interval.

    This is a geometry metric, not yet an IR detection metric.
    """
    rows = []

    for neo_class in NEO_FILES:
        group = summary[summary["neo_class"] == neo_class]
        pha_group = group[group["PHA"] == True]

        for threshold in THRESHOLDS_DEG:
            enter = group["min_elong_deg"] <= threshold
            pha_enter = pha_group["min_elong_deg"] <= threshold

            rows.append({
                "neo_class": neo_class,
                "threshold_deg": threshold,
                "objects": len(group),
                "objects_entering": int(enter.sum()),
                "fraction_entering_percent": (
                    100.0 * enter.mean()
                    if len(group) else np.nan
                ),
                "PHA_objects": len(pha_group),
                "PHA_objects_entering": int(pha_enter.sum()),
                "PHA_fraction_entering_percent": (
                    100.0 * pha_enter.mean()
                    if len(pha_group) else np.nan
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_elongation_distribution(
    histogram_table: pd.DataFrame,
    output_dir: Path,
    pha_only: bool,
):
    """
    Plot normalized object-time sampling versus solar elongation.

    Each class is normalized independently so Apollo's much larger population
    does not visually dominate the smaller classes.
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    value_col = (
        "PHA_sample_fraction_percent"
        if pha_only
        else "all_sample_fraction_percent"
    )

    for neo_class in NEO_FILES:
        group = histogram_table[
            histogram_table["neo_class"] == neo_class
        ]

        ax.plot(
            group["bin_center_deg"],
            group[value_col],
            marker="o",
            markersize=3,
            label=neo_class,
        )

    ax.set_xlim(0, 60)
    ax.set_xlabel("Solar elongation (deg)")
    ax.set_ylabel("Fraction of object-time samples per 5° bin (%)")

    if pha_only:
        ax.set_title("Solar elongation distribution — PHA subset")
        filename = "solar_elongation_distribution_PHA.png"
    else:
        ax.set_title("Solar elongation distribution — all NEOs")
        filename = "solar_elongation_distribution_all.png"

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_minimum_elongation(
    summary: pd.DataFrame,
    output_dir: Path,
    pha_only: bool,
):
    """Histogram of the minimum elongation reached by each unique object."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bins = np.arange(0, 65, 5)

    for neo_class in NEO_FILES:
        group = summary[summary["neo_class"] == neo_class]

        if pha_only:
            group = group[group["PHA"] == True]

        values = group["min_elong_deg"].dropna()

        if len(values):
            ax.hist(
                values,
                bins=bins,
                histtype="step",
                linewidth=1.6,
                label=f"{neo_class} (N={len(values)})",
            )

    ax.set_xlim(0, 60)
    ax.set_xlabel("Minimum solar elongation reached (deg)")
    ax.set_ylabel("Number of unique asteroids")

    if pha_only:
        ax.set_title("Minimum solar elongation — PHA subset")
        filename = "minimum_solar_elongation_PHA.png"
    else:
        ax.set_title("Minimum solar elongation — all NEOs")
        filename = "minimum_solar_elongation_all.png"

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


def plot_threshold_entry(
    threshold_table: pd.DataFrame,
    output_dir: Path,
    pha_only: bool,
):
    """
    Plot the fraction of unique objects that enter within an elongation angle.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    y_col = (
        "PHA_fraction_entering_percent"
        if pha_only
        else "fraction_entering_percent"
    )

    for neo_class in NEO_FILES:
        group = threshold_table[
            threshold_table["neo_class"] == neo_class
        ]

        ax.plot(
            group["threshold_deg"],
            group[y_col],
            marker="o",
            label=neo_class,
        )

    ax.set_xlabel("Solar elongation threshold (deg)")
    ax.set_ylabel("Objects entering within threshold (%)")
    ax.set_ylim(0, 100)

    if pha_only:
        ax.set_title("PHA population entering low-solar-elongation regions")
        filename = "low_elongation_entry_fraction_PHA.png"
    else:
        ax.set_title("NEO population entering low-solar-elongation regions")
        filename = "low_elongation_entry_fraction_all.png"

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Propagate cleaned NEO catalogs and calculate Earth-centered "
            "solar elongation."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("cleaned data"),
        help="Folder containing the four cleaned NEO CSV files.",
    )

    parser.add_argument(
    "--earth",
    type=Path,
    default=Path("cleaned data") / "earth_ephemeris.csv",
    help="Cleaned JPL Horizons Earth ephemeris CSV.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("solar elongation results"),
        help="Directory for output CSVs and figures.",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Number of asteroids propagated at once.",
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    earth = load_earth_ephemeris(args.earth)

    print(
        f"Earth ephemeris: {len(earth)} daily states "
        f"from JD {earth['jd_tdb'].iloc[0]:.1f} "
        f"to {earth['jd_tdb'].iloc[-1]:.1f}"
    )

    all_summaries = []
    hist_by_class = {}
    skipped_parts = []

    for neo_class, filename in NEO_FILES.items():
        path = args.data_dir / filename

        print(f"\nLoading {neo_class}: {path}")

        catalog = load_neo_catalog(path, neo_class)

        summary, histograms, skipped = analyze_class(
            catalog,
            earth,
            neo_class,
            args.chunk_size,
        )

        all_summaries.append(summary)
        hist_by_class[neo_class] = histograms
        skipped_parts.extend(skipped)

    # ------------------------------------------------------------------
    # Combined per-object summary
    # ------------------------------------------------------------------
    elongation_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    elongation_summary.to_csv(
        args.output_dir / "solar_elongation_object_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Aggregate histogram
    # ------------------------------------------------------------------
    histogram_table = make_histogram_table(hist_by_class)
    histogram_table.to_csv(
        args.output_dir / "solar_elongation_histogram.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Unique-object low-angle entry statistics
    # ------------------------------------------------------------------
    threshold_table = make_threshold_table(elongation_summary)
    threshold_table.to_csv(
        args.output_dir / "low_elongation_threshold_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Skipped rows
    # ------------------------------------------------------------------
    skipped_df = pd.concat(skipped_parts, ignore_index=True)
    skipped_df.to_csv(
        args.output_dir / "skipped_objects.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    plot_elongation_distribution(
        histogram_table,
        args.output_dir,
        pha_only=False,
    )
    plot_elongation_distribution(
        histogram_table,
        args.output_dir,
        pha_only=True,
    )

    plot_minimum_elongation(
        elongation_summary,
        args.output_dir,
        pha_only=False,
    )
    plot_minimum_elongation(
        elongation_summary,
        args.output_dir,
        pha_only=True,
    )

    plot_threshold_entry(
        threshold_table,
        args.output_dir,
        pha_only=False,
    )
    plot_threshold_entry(
        threshold_table,
        args.output_dir,
        pha_only=True,
    )

    print("\nDone.")
    print(f"Results saved in: {args.output_dir.resolve()}")
    print(f"Objects propagated: {len(elongation_summary)}")
    print(f"Objects skipped   : {len(skipped_df)}")


if __name__ == "__main__":
    main()

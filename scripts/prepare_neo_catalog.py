#!/usr/bin/env python3
"""
prepare_neo_catalog_clean.py

Clean a JPL SBDB asteroid-class CSV and save only mission-relevant fields.

Expected JPL input fields:
full_name, a, e, i, om, w, q, ad, per_y,
data_arc, condition_code, n_obs_used, n_del_obs_used, n_dop_obs_used,
H, spkid, pha, diameter, albedo, diameter_sigma,
epoch, ma, moid
"""

import argparse
from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "full_name", "a", "e", "i", "om", "w", "q", "ad", "per_y",
    "data_arc", "condition_code", "n_obs_used", "n_del_obs_used",
    "n_dop_obs_used", "H", "spkid", "pha", "diameter", "albedo",
    "diameter_sigma", "epoch", "moid",
]

RENAME_MAP = {
    "full_name": "name",
    "a": "a_au",
    "e": "e",
    "i": "i_deg",
    "om": "Omega_deg",
    "w": "omega_deg",
    "q": "q_au",
    "ad": "Q_au",
    "per_y": "period_years",
    "epoch": "epoch_jd",
    "ma": "M_deg",
    "M": "M_deg",
    "moid": "moid_au",
    "H": "H",
    "spkid": "spkid",
    "pha": "PHA",
    "diameter": "diameter_km",
    "diameter_sigma": "diameter_sigma_km",
    "albedo": "albedo",
    "data_arc": "data_arc_days",
    "condition_code": "condition_code",
    "n_obs_used": "n_obs_used",
    "n_del_obs_used": "n_delay_obs_used",
    "n_dop_obs_used": "n_doppler_obs_used",
}

NUMERIC_COLUMNS = [
    "a_au", "e", "i_deg", "Omega_deg", "omega_deg", "q_au", "Q_au",
    "period_years", "epoch_jd", "M_deg", "moid_au", "H", "diameter_km",
    "diameter_sigma_km", "albedo", "data_arc_days", "condition_code",
    "n_obs_used", "n_delay_obs_used", "n_doppler_obs_used",
]

OUTPUT_COLUMNS = [
    "name", "spkid",
    "a_au", "e", "i_deg", "Omega_deg", "omega_deg", "q_au", "Q_au",
    "period_years", "period_days", "epoch_jd", "M_deg",
    "moid_au", "PHA",
    "H", "diameter_km", "diameter_sigma_km", "albedo",
    "data_arc_days", "condition_code", "n_obs_used",
    "n_delay_obs_used", "n_doppler_obs_used",
]


def clean_pha(value):
    if pd.isna(value):
        return pd.NA

    value = str(value).strip().upper()

    if value == "Y":
        return True
    if value == "N":
        return False

    return pd.NA


def prepare_catalog(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        input_file,
        na_values=["", " ", "null", "NULL", "None", "nan", "NaN"],
    )

    df.columns = [
        str(col).replace("\ufeff", "").strip().strip('"')
        for col in df.columns
    ]

    if "ma" in df.columns:
        mean_anomaly_column = "ma"
    elif "M" in df.columns:
        mean_anomaly_column = "M"
    else:
        raise ValueError("Missing mean-anomaly column: expected 'ma' or 'M'.")

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    columns_to_keep = list(dict.fromkeys(REQUIRED_COLUMNS + [mean_anomaly_column]))
    df = df[columns_to_keep].copy()
    df = df.rename(columns=RENAME_MAP)

    df["name"] = df["name"].astype("string").str.strip()
    df["spkid"] = df["spkid"].astype("string").str.strip()

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["PHA"] = df["PHA"].apply(clean_pha).astype("boolean")
    df["period_days"] = df["period_years"] * 365.25

    return df[OUTPUT_COLUMNS]


def main():
    parser = argparse.ArgumentParser(
        description="Clean a JPL SBDB NEO catalog for mission analysis."
    )
    parser.add_argument("input_csv", type=Path, help="Path to the raw JPL CSV.")
    args = parser.parse_args()

    input_file = args.input_csv.resolve()

    output_dir = input_file.parent / "cleaned data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{input_file.stem}_cleaned.csv"

    cleaned = prepare_catalog(input_file)
    cleaned.to_csv(output_file, index=False)

    pha_count = int(cleaned["PHA"].fillna(False).sum())

    print(f"Processed objects : {len(cleaned)}")
    print(f"PHA objects       : {pha_count}")
    print(f"Saved             : {output_file}")


if __name__ == "__main__":
    main()

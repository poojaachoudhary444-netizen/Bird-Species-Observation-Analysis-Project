"""
app.py — Bird Species Observation Analysis: Data Cleaning & Loading Pipeline
==============================================================================
Reads the raw multi-sheet Forest & Grassland observation workbooks, cleans
and consolidates them into a single tidy table, engineers a handful of
analysis-ready features, and persists the result to:
  1. data/bird_data_cleaned.csv   (flat file, easy to inspect / share)
  2. db/bird_observations.db      (SQLite database, table: observations)

This is the script the notebook and the Streamlit dashboard both build on.
Run it directly whenever the source data changes:

    python app.py

--------------------------------------------------------------------------
Swapping in the REAL dataset
--------------------------------------------------------------------------
Replace the two files in data/ with the real workbooks (same sheet-per-
admin-unit layout described in the project brief), keeping these names —
or just change RAW_FOREST_PATH / RAW_GRASSLAND_PATH below:

    data/Bird_Monitoring_Data_FOREST.xlsx
    data/Bird_Monitoring_Data_GRASSLAND.xlsx

Then re-run `python app.py`. No other code changes are required as long as
column names match the schema in the project brief.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(exist_ok=True)

RAW_FOREST_PATH = DATA_DIR / "Bird_Monitoring_Data_FOREST.xlsx"
RAW_GRASSLAND_PATH = DATA_DIR / "Bird_Monitoring_Data_GRASSLAND.xlsx"

CLEANED_CSV_PATH = DATA_DIR / "bird_data_cleaned.csv"
SQLITE_DB_PATH = DB_DIR / "bird_observations.db"
TABLE_NAME = "observations"

# Columns expected per the project brief
EXPECTED_COLUMNS = [
    "Admin_Unit_Code", "Sub_Unit_Code", "Site_Name", "Plot_Name", "Location_Type",
    "Year", "Date", "Start_Time", "End_Time", "Observer", "Visit",
    "Interval_Length", "ID_Method", "Distance", "Flyover_Observed", "Sex",
    "Common_Name", "Scientific_Name", "AcceptedTSN", "NPSTaxonCode", "AOU_Code",
    "PIF_Watchlist_Status", "Regional_Stewardship_Status", "Temperature",
    "Humidity", "Sky", "Wind", "Disturbance", "Initial_Three_Min_Cnt",
]

ADMIN_UNIT_NAMES = {
    "ANTI": "Antietam National Battlefield",
    "CATO": "Catoctin Mountain Park",
    "CHOH": "Chesapeake & Ohio Canal NHP",
    "GWMP": "George Washington Memorial Parkway",
    "HAFE": "Harpers Ferry NHP",
    "MANA": "Manassas National Battlefield Park",
    "MONO": "Monocacy National Battlefield",
    "NACE": "National Capital East Parks",
    "PRWI": "Prince William Forest Park",
    "ROCR": "Rock Creek Park",
    "WOTR": "Wolf Trap National Park for the Performing Arts",
}


# --------------------------------------------------------------------------
# 1. Read multi-sheet workbooks
# --------------------------------------------------------------------------
def read_multi_sheet_workbook(path: Path, location_type_fallback: str) -> pd.DataFrame:
    """Read every sheet (one per Admin_Unit_Code) from a workbook and stack them."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run generate_sample_data.py for a demo dataset, "
            f"or place the real workbook at this path."
        )
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    frames = []
    for sheet_name, df in sheets.items():
        df = df.copy()
        if "Admin_Unit_Code" not in df.columns:
            df["Admin_Unit_Code"] = sheet_name
        if "Location_Type" not in df.columns:
            df["Location_Type"] = location_type_fallback
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_raw_data() -> pd.DataFrame:
    forest_df = read_multi_sheet_workbook(RAW_FOREST_PATH, "Forest")
    grassland_df = read_multi_sheet_workbook(RAW_GRASSLAND_PATH, "Grassland")
    combined = pd.concat([forest_df, grassland_df], ignore_index=True)
    return combined


# --------------------------------------------------------------------------
# 2. Clean & standardize
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Standardize column names (strip whitespace, keep the documented names) ---
    df.columns = [c.strip() for c in df.columns]

    # --- Keep only expected columns that exist, warn about the rest silently kept ---
    missing_expected = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_expected:
        for c in missing_expected:
            df[c] = np.nan  # ensure schema consistency even if a source lacks a column

    # --- Drop exact duplicate rows ---
    before = len(df)
    df = df.drop_duplicates()
    dupes_removed = before - len(df)

    # --- Parse dates & times ---
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(df["Date"].dt.year)
    df["Year"] = df["Year"].astype("Int64")

    def _parse_time(series):
        return pd.to_datetime(series.astype(str), format="%H:%M", errors="coerce").dt.time

    df["Start_Time"] = _parse_time(df["Start_Time"])
    df["End_Time"] = _parse_time(df["End_Time"])

    # --- Standardize text/categorical fields ---
    text_cols = [
        "Admin_Unit_Code", "Sub_Unit_Code", "Site_Name", "Plot_Name", "Location_Type",
        "Observer", "Interval_Length", "ID_Method", "Distance", "Sex", "Common_Name",
        "Scientific_Name", "AOU_Code", "Sky", "Wind", "Disturbance",
    ]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    if "Location_Type" in df.columns:
        df["Location_Type"] = df["Location_Type"].str.title()

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].str.title().replace(
            {"U": "Undetermined", "M": "Male", "F": "Female"}
        )
        df["Sex"] = df["Sex"].fillna("Undetermined")

    # --- Booleans stored inconsistently as TRUE/FALSE/1/0/strings ---
    bool_cols = ["Flyover_Observed", "PIF_Watchlist_Status", "Regional_Stewardship_Status"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .str.upper()
                .map({"TRUE": True, "FALSE": False, "1": True, "0": False, "YES": True, "NO": False})
            )
            df[c] = df[c].fillna(False)

    # --- Numeric fields ---
    numeric_cols = ["Temperature", "Humidity", "Initial_Three_Min_Cnt", "Visit",
                     "AcceptedTSN"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Impute a small number of missing numeric values with the group (species) median
    for c in ["Temperature", "Humidity"]:
        if c in df.columns:
            df[c] = df.groupby("Common_Name")[c].transform(lambda s: s.fillna(s.median()))
            df[c] = df[c].fillna(df[c].median())

    if "Initial_Three_Min_Cnt" in df.columns:
        df["Initial_Three_Min_Cnt"] = df["Initial_Three_Min_Cnt"].fillna(1).astype(int)

    # --- Drop rows with no species identified (core to every analysis) ---
    before = len(df)
    df = df.dropna(subset=["Common_Name", "Scientific_Name"])
    species_dropped = before - len(df)

    # --- Admin unit full name lookup ---
    df["Admin_Unit_Name"] = df["Admin_Unit_Code"].map(ADMIN_UNIT_NAMES).fillna(df["Admin_Unit_Code"])

    # --------------------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------------------
    df["Month"] = df["Date"].dt.month
    df["Month_Name"] = df["Date"].dt.strftime("%b")

    season_map = {12: "Winter", 1: "Winter", 2: "Winter",
                  3: "Spring", 4: "Spring", 5: "Spring",
                  6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Fall", 10: "Fall", 11: "Fall"}
    df["Season"] = df["Month"].map(season_map)

    def _hour_bin(t):
        if pd.isna(t):
            return np.nan
        h = t.hour
        if h < 7:
            return "Early Morning (<7am)"
        elif h < 9:
            return "Morning (7-9am)"
        elif h < 11:
            return "Late Morning (9-11am)"
        else:
            return "Midday+ (11am+)"

    df["Observation_Time_Bin"] = df["Start_Time"].apply(_hour_bin)

    # Approximate numeric distance (meters) from the categorical bucket, for correlation use
    distance_map = {
        "<= 50 Meters": 25,
        "50 - 100 Meters": 75,
        "> 100 Meters": 125,
    }
    df["Distance_Approx_m"] = df["Distance"].map(distance_map)

    df["Watchlist_Flag"] = np.where(df["PIF_Watchlist_Status"], "At-Risk (Watchlist)", "Not Listed")

    df = df.reset_index(drop=True)

    print(f"Cleaning summary: removed {dupes_removed} duplicate rows, "
          f"{species_dropped} rows missing species ID. Final row count: {len(df):,}")

    return df


# --------------------------------------------------------------------------
# 3. Persist
# --------------------------------------------------------------------------
def save_outputs(df: pd.DataFrame):
    # Flat file
    out = df.copy()
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    out["Start_Time"] = out["Start_Time"].astype(str)
    out["End_Time"] = out["End_Time"].astype(str)
    out.to_csv(CLEANED_CSV_PATH, index=False)
    print(f"Saved cleaned CSV -> {CLEANED_CSV_PATH} ({len(out):,} rows)")

    # SQLite
    conn = sqlite3.connect(SQLITE_DB_PATH)
    out.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_location ON {TABLE_NAME}(Location_Type)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_species ON {TABLE_NAME}(Common_Name)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_year ON {TABLE_NAME}(Year)")
    conn.commit()
    conn.close()
    print(f"Saved SQLite DB -> {SQLITE_DB_PATH} (table: {TABLE_NAME})")


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def run_pipeline() -> pd.DataFrame:
    print("Loading raw workbooks...")
    raw_df = load_raw_data()
    print(f"Raw rows loaded: {len(raw_df):,} across "
          f"{raw_df['Admin_Unit_Code'].nunique()} admin units and "
          f"{raw_df['Location_Type'].nunique()} habitat types.")

    print("Cleaning & engineering features...")
    clean_df = clean_data(raw_df)

    print("Saving outputs...")
    save_outputs(clean_df)

    return clean_df


if __name__ == "__main__":
    run_pipeline()
    print("Pipeline complete.")

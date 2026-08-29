"""
generate_sample_data.py
------------------------
Creates SYNTHETIC sample data that matches the schema described in the
"Bird Species Observation Analysis" project brief:
  - Two workbooks: Bird_Monitoring_Data_FOREST.XLSX and Bird_Monitoring_Data_GRASSLAND.XLSX
  - Each workbook has one sheet per Admin_Unit_Code:
    ANTI, CATO, CHOH, GWMP, HAFE, MANA, MONO, NACE, PRWI, ROCR, WOTR

This lets app.py / the notebook / the dashboard run end-to-end immediately.
Once you have the REAL "Bird_Observation_DataSet" workbook(s), just drop them
into the data/ folder with the same file names (or update RAW_FOREST_PATH /
RAW_GRASSLAND_PATH in app.py) and re-run the pipeline — no other code changes
needed.

NOTE: This data is randomly generated for demo/testing purposes only and does
not represent real bird observations.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

ADMIN_UNITS = ["ANTI", "CATO", "CHOH", "GWMP", "HAFE", "MANA", "MONO", "NACE", "PRWI", "ROCR", "WOTR"]

SPECIES = [
    # (Common_Name, Scientific_Name, AOU_Code, AcceptedTSN, NPSTaxonCode, PIF_Watchlist, Regional_Stewardship)
    ("Eastern Towhee", "Pipilo erythrophthalmus", "EATO", 179269, "NPS-EATO", True, True),
    ("Northern Cardinal", "Cardinalis cardinalis", "NOCA", 179263, "NPS-NOCA", False, False),
    ("Carolina Wren", "Thryothorus ludovicianus", "CARW", 178447, "NPS-CARW", False, True),
    ("Wood Thrush", "Hylocichla mustelina", "WOTH", 178353, "NPS-WOTH", True, True),
    ("American Robin", "Turdus migratorius", "AMRO", 178396, "NPS-AMRO", False, False),
    ("Blue Jay", "Cyanocitta cristata", "BLJA", 179738, "NPS-BLJA", False, False),
    ("Red-winged Blackbird", "Agelaius phoeniceus", "RWBL", 179466, "NPS-RWBL", False, False),
    ("Field Sparrow", "Spizella pusilla", "FISP", 179218, "NPS-FISP", True, True),
    ("Indigo Bunting", "Passerina cyanea", "INBU", 179328, "NPS-INBU", False, True),
    ("Eastern Meadowlark", "Sturnella magna", "EAME", 179545, "NPS-EAME", True, True),
    ("Grasshopper Sparrow", "Ammodramus savannarum", "GRSP", 179226, "NPS-GRSP", True, True),
    ("Barn Swallow", "Hirundo rustica", "BARS", 178576, "NPS-BARS", False, False),
    ("American Goldfinch", "Spinus tristis", "AMGO", 179243, "NPS-AMGO", False, False),
    ("Yellow Warbler", "Setophaga petechia", "YEWA", 178911, "NPS-YEWA", False, False),
    ("Scarlet Tanager", "Piranga olivacea", "SCTA", 179317, "NPS-SCTA", True, False),
    ("Downy Woodpecker", "Dryobates pubescens", "DOWO", 178196, "NPS-DOWO", False, False),
    ("Tufted Titmouse", "Baeolophus bicolor", "TUTI", 178420, "NPS-TUTI", False, False),
    ("Song Sparrow", "Melospiza melodia", "SOSP", 179231, "NPS-SOSP", False, False),
    ("Red-tailed Hawk", "Buteo jamaicensis", "RTHA", 175350, "NPS-RTHA", False, False),
    ("Bobolink", "Dolichonyx oryzivorus", "BOBO", 179462, "NPS-BOBO", True, True),
]

OBSERVERS = ["A. Rivera", "J. Chen", "M. Okafor", "S. Patel", "T. Nguyen", "R. Fischer", "L. Martin", "K. Osei"]
ID_METHODS = ["Singing", "Calling", "Visualization", "Non-vocal"]
INTERVALS = ["0-2.5 min", "2.5-5 min", "5-7.5 min", "7.5-10 min"]
DISTANCES = ["<= 50 Meters", "50 - 100 Meters", "> 100 Meters"]
SKY = ["Clear", "Partly Cloudy", "Cloudy/Overcast", "Fog/Mist", "Light Rain"]
WIND = ["Calm (< 1 mph) smoke rises vertically", "Light air (1-3 mph)", "Light breeze (4-7 mph)", "Gentle breeze (8-12 mph)"]
DISTURBANCE = ["No effect on count", "Slight effect on count", "Moderate effect on count"]
SEX = ["Male", "Female", "Undetermined"]


def _random_times():
    start_hour = RNG.integers(5, 10)
    start_min = RNG.choice([0, 10, 15, 20, 30, 40, 45, 50])
    start = pd.Timestamp(2023, 1, 1, int(start_hour), int(start_min))
    end = start + pd.Timedelta(minutes=int(RNG.integers(8, 12)))
    return start.strftime("%H:%M"), end.strftime("%H:%M")


def _make_sheet(admin_unit: str, location_type: str, n_rows: int) -> pd.DataFrame:
    rows = []
    n_plots = RNG.integers(4, 9)
    plot_names = [f"{admin_unit}-P{p:02d}" for p in range(1, n_plots + 1)]
    n_sites = RNG.integers(2, 5)
    site_names = [f"{admin_unit} Site {s}" for s in range(1, n_sites + 1)]

    for i in range(n_rows):
        year = int(RNG.choice([2021, 2022, 2023, 2024]))
        month = int(RNG.integers(4, 8))  # spring/summer survey season
        day = int(RNG.integers(1, 28))
        date = pd.Timestamp(year, month, day)
        start_time, end_time = _random_times()

        species = SPECIES[RNG.integers(0, len(SPECIES))]
        common_name, sci_name, aou, tsn, npstc, pif_watch, reg_steward = species

        # base temperature/humidity vary a bit by habitat type for a realistic signal
        base_temp = 68 if location_type == "Forest" else 74
        temperature = round(float(RNG.normal(base_temp, 8)), 1)
        humidity = int(np.clip(RNG.normal(60, 15), 20, 100))

        flyover = bool(RNG.random() < (0.12 if location_type == "Forest" else 0.22))
        initial_cnt = int(RNG.poisson(2) + 1)

        rows.append(
            {
                "Admin_Unit_Code": admin_unit,
                "Sub_Unit_Code": f"{admin_unit}-{RNG.integers(1,4)}",
                "Site_Name": site_names[RNG.integers(0, len(site_names))],
                "Plot_Name": plot_names[RNG.integers(0, len(plot_names))],
                "Location_Type": location_type,
                "Year": year,
                "Date": date,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Observer": OBSERVERS[RNG.integers(0, len(OBSERVERS))],
                "Visit": int(RNG.integers(1, 4)),
                "Interval_Length": INTERVALS[RNG.integers(0, len(INTERVALS))],
                "ID_Method": ID_METHODS[RNG.integers(0, len(ID_METHODS))],
                "Distance": DISTANCES[RNG.integers(0, len(DISTANCES))],
                "Flyover_Observed": flyover,
                "Sex": SEX[RNG.integers(0, len(SEX))],
                "Common_Name": common_name,
                "Scientific_Name": sci_name,
                "AcceptedTSN": tsn,
                "NPSTaxonCode": npstc,
                "AOU_Code": aou,
                "PIF_Watchlist_Status": pif_watch,
                "Regional_Stewardship_Status": reg_steward,
                "Temperature": temperature,
                "Humidity": humidity,
                "Sky": SKY[RNG.integers(0, len(SKY))],
                "Wind": WIND[RNG.integers(0, len(WIND))],
                "Disturbance": DISTURBANCE[RNG.integers(0, len(DISTURBANCE))],
                "Initial_Three_Min_Cnt": initial_cnt,
            }
        )

    df = pd.DataFrame(rows)

    # sprinkle a few missing values / duplicates so the cleaning step has real work to do
    for col in ["Temperature", "Humidity", "Sex", "Distance"]:
        mask = RNG.random(len(df)) < 0.02
        df.loc[mask, col] = np.nan
    if len(df) > 5:
        df = pd.concat([df, df.sample(3, random_state=1)], ignore_index=True)

    return df


def build_workbook(location_type: str, path: Path, rows_per_unit=(150, 400)):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for unit in ADMIN_UNITS:
            n = int(RNG.integers(*rows_per_unit))
            sheet_df = _make_sheet(unit, location_type, n)
            sheet_df.to_excel(writer, sheet_name=unit, index=False)
    print(f"Wrote {path} ({location_type})")


def generate():
    """Callable entry point (safe to import and call explicitly, unlike
    relying on __main__ side effects) — used by app.py / dashboard.py to
    self-heal when raw data is missing."""
    build_workbook("Forest", DATA_DIR / "Bird_Monitoring_Data_FOREST.xlsx")
    build_workbook("Grassland", DATA_DIR / "Bird_Monitoring_Data_GRASSLAND.xlsx")
    print("Sample data generation complete.")


if __name__ == "__main__":
    generate()

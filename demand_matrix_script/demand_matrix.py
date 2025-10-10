"""
Demand matrix calculator (gravity model).

Assumes trip generation CSV file has columns:
- centroid_id
- productions
- attractions

Assumes free flow time CSV file has columns:
- row (origin zone)
- column (destination zone)
- free_flow_time (seconds)
"""

import os
import numpy as np
import pandas as pd

INTRA_ZONAL_SECONDS = 90.0
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRIPS_CSV_PATH = os.path.join(SCRIPT_DIR, "trip_generation.csv")
IMPEDANCE_CSV_PATH = os.path.join(SCRIPT_DIR, "free_flow_time.csv")
OUTPUT_CSV_PATH = os.path.join(SCRIPT_DIR, "demand_matrix.csv")


def load_trip_generation(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    for col in ("productions", "attractions"):
        df[col] = pd.to_numeric(df[col])

    result = df[["centroid_id", "productions", "attractions"]]
    print("Trip generation matrix:")
    print(result)
    return result


def load_free_flow_time(csv_path: str, centroid_ids: list[int], intra_seconds: float) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["row"] = pd.to_numeric(df["row"])
    df["column"] = pd.to_numeric(df["column"])
    df["free_flow_time"] = pd.to_numeric(df["free_flow_time"])

    F = (
        df.pivot(index="row", columns="column", values="free_flow_time")
          .reindex(index=centroid_ids, columns=centroid_ids)
    )

    # intra-zonal values on the diagonal
    np.fill_diagonal(F.values, intra_seconds)

    F.index.name = "ZoneOrig"
    F.columns.name = "ZoneDest"
    print("Impedance matrix:")
    print(F)
    return F


def compute_demand_matrix(trips: pd.DataFrame, F_time: pd.DataFrame) -> pd.DataFrame:
    productions = trips.set_index("centroid_id")["productions"].astype(float)
    attractions = trips.set_index("centroid_id")["attractions"].astype(float)

    F_imp = 1.0 / (F_time ** 2)

    AF = F_imp.mul(attractions, axis=1)
    denom = AF.sum(axis=1)

    T = AF.mul(productions, axis=0).div(denom, axis=0)
    T.index.name = "ZoneOrig"
    T.columns.name = "ZoneDest"
    return T

def save_demand_matrix(T: pd.DataFrame, csv_path: str):
    T_long = T.stack().reset_index(name="Demand")
    T_long.to_csv(csv_path, index=False)


def main():
    trips = load_trip_generation(TRIPS_CSV_PATH)
    zones = trips["centroid_id"].tolist()

    F_time = load_free_flow_time(IMPEDANCE_CSV_PATH, zones, INTRA_ZONAL_SECONDS)
    T = compute_demand_matrix(trips, F_time)
    T = T.round().astype(int)  # round to nearest integer

    save_demand_matrix(T, OUTPUT_CSV_PATH)
    print(f"Saved {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
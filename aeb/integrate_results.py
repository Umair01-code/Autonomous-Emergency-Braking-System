
from pathlib import Path

import numpy as np
import pandas as pd

from aeb.risk import assess_risk


CSV_PATH = Path("data/AEB_results.csv")
OUTPUT_PATH = Path("data/AEB_integrated_results.csv")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {CSV_PATH.resolve()}"
        )

    df = pd.read_csv(CSV_PATH)

    print("CSV columns:", df.columns.tolist())

    # Prefer the smoothed TTC if it exists.
    ttc_column = next(
        (
            col for col in ("ttc_smooth", "ttc")
            if col in df.columns
        ),
        None,
    )

    if ttc_column is None:
        raise ValueError(
            "No TTC column found. We need to inspect your CSV "
            "and calculate TTC from distance and relative speed."
        )

    df["ttc_used"] = pd.to_numeric(
        df[ttc_column], errors="coerce"
    )

    df["aeb_decision"] = df["ttc_used"].apply(
        lambda ttc: assess_risk(ttc)
        if pd.notna(ttc) and ttc >= 0
        else "UNKNOWN"
    )

    df["aeb_active"] = df["aeb_decision"].eq("AEB")

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nAEB decision counts:")
    print(df["aeb_decision"].value_counts(dropna=False))

    print(f"\nSaved integrated results to: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

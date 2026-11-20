
from pathlib import Path
import pandas as pd

csv_path = Path("data/AEB_integrated_results.csv")
video_path = Path("runs/track/aeb_ttc_output.mp4")

if not csv_path.exists():
    raise FileNotFoundError(f"Missing CSV: {csv_path}")

df = pd.read_csv(csv_path)

print("========== AEB VALIDATION ==========")
print("CSV rows:", len(df))
print("Video exists:", video_path.exists())

if video_path.exists():
    print("Video size (MB):", round(
        video_path.stat().st_size / (1024 * 1024), 2
    ))

print("\nRisk counts:")
print(df["aeb_decision"].value_counts(dropna=False))

print("\nTTC statistics:")
print(df["ttc_used"].describe())

print("\nMissing TTC values:", df["ttc_used"].isna().sum())
print("Unique tracked vehicles:", df["track_id"].nunique())

print("\nRows with AEB decision:")
print(
    df.loc[
        df["aeb_decision"] == "AEB",
        ["frame", "track_id", "ttc_used", "aeb_decision"]
    ].head(10).to_string(index=False)
)

print("\nValidation report completed.")

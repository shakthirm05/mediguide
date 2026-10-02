from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE = (
    PROJECT_ROOT
    / "data"
    / "hospital"
    / "mediguide_departments_final_updated.csv"
)

print("=" * 70)
print("LOCATION DATA VERIFICATION")
print("=" * 70)

# Load CSV
departments = pd.read_csv(FILE)

print("\nColumns found:")
print(departments.columns.tolist())
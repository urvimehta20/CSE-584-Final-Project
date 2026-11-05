"""Inspect ClimRetrieve source files. Do not assume column names."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "ClimRetrieve"

CANDIDATES = [
    RAW / "Report-Level Dataset" / "ClimRetrieve_ReportLevel_V1.csv",
    RAW / "Report-Level Dataset" / "Intermediate_Steps_Data" / "report_level_data_final_test.csv",
    RAW / "Report-Level Dataset" / "Intermediate_Steps_Data" / "report_level_data_test.csv",
    RAW / "Expert-Annotated Relevant Sources Dataset" / "ClimRetrieve_base.xlsx",
    RAW / "Expert-Annotated Relevant Sources Dataset" / "Core_Questions_with_Explanations.xlsx",
]


def _read(path: Path) -> list[tuple[str, pd.DataFrame]]:
    if path.suffix.lower() == ".csv":
        return [("csv", pd.read_csv(path))]
    if path.suffix.lower() in {".xlsx", ".xls"}:
        xl = pd.ExcelFile(path)
        return [(sheet, xl.parse(sheet)) for sheet in xl.sheet_names]
    raise ValueError(f"Unsupported file: {path}")


def inspect_one(path: Path) -> dict:
    out: dict = {
        "path": str(path),
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.exists() else None,
        "tables": [],
    }
    if not path.exists():
        print(f"\n=== MISSING: {path} ===")
        return out

    print(f"\n{'=' * 80}")
    print(f"PATH: {path}")
    print(f"SIZE_BYTES: {path.stat().st_size}")

    for table_name, df in _read(path):
        sample = df.head(3)
        table = {
            "table": table_name,
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "columns": [str(c) for c in df.columns],
            "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
            "null_counts": {str(c): int(df[c].isna().sum()) for c in df.columns},
            "sample_rows": json.loads(sample.to_json(orient="records", date_format="iso")),
        }
        out["tables"].append(table)
        print(f"\nTABLE: {table_name}")
        print(f"ROWS: {len(df)}")
        print(f"COLUMNS ({df.shape[1]}): {list(df.columns)}")
        print("DTYPES:")
        for col, dtype in df.dtypes.items():
            print(f"  - {col}: {dtype}")
        print("NULL_COUNTS:")
        for col, nnull in table["null_counts"].items():
            print(f"  - {col}: {nnull}")
        print("SAMPLE_3_ROWS:")
        pd.set_option("display.max_colwidth", 160)
        pd.set_option("display.width", 200)
        print(sample.to_string(index=False))
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            print(f"NUNIQUE {col}: {nunique}")
    return out


def main() -> None:
    reports: list[dict] = []
    for path in CANDIDATES:
        reports.append(inspect_one(path))

    out_path = Path(__file__).resolve().parents[1] / "results" / "inspect_source_files.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(reports, indent=2, default=str))
    print(f"\nWrote inspection JSON to {out_path}")


if __name__ == "__main__":
    main()

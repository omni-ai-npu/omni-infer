# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Formula:
    left: str
    right: str
    output: str | None = None


# Keep in sync with parse_logs._ENCODE_ACTION_KEYS (time_analysis column names for encode).
_ENCODE_TRACE_KEYS = frozenset(
    {
        "Encoder api server get request",
        "Finish process request for encode engine",
        "Start process request in encode engine",
        "Encoder add waiting queue",
        "Encoder try to schedule in waiting queue",
        "Encoder start has_caches",
        "Encoder done has_caches",
        "Start append running sequece for encode",
        "Encoder start execute_model",
        "Encoder start _execute_mm_encoder",
        "Encoder start save_caches",
        "Encoder done save_caches",
        "Encoder done _execute_mm_encoder",
        "Encoder done execute_model",
        "Finish encode pickle and start response",
    }
)


def resolve_rules_path() -> Path:
    return (Path(__file__).resolve().parent / "time_analysis_rules.yaml").resolve()


def parse_formula_file(path: Path) -> list[Formula]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Rules file must be a YAML mapping: {path}")

    formulas_data = data.get("formulas")
    if not isinstance(formulas_data, list) or not formulas_data:
        raise ValueError(f"'formulas' must be a non-empty list in: {path}")

    formulas: list[Formula] = []
    for index, item in enumerate(formulas_data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Formula #{index} must be a mapping")
        left = str(item.get("left", "")).strip()
        right = str(item.get("right", "")).strip()
        output = item.get("output")
        if not left or not right:
            raise ValueError(f"Formula #{index} must contain non-empty 'left' and 'right'")
        if output is not None:
            output = str(output).strip()
            if not output:
                output = None
        formulas.append(Formula(left=left, right=right, output=output))
    return formulas


def _normalize_key(s: str) -> str:
    return " ".join(str(s).strip().split())


def _formula_uses_encode_column(formula: Formula) -> bool:
    enc = {_normalize_key(k) for k in _ENCODE_TRACE_KEYS}
    return _normalize_key(formula.left) in enc or _normalize_key(formula.right) in enc


def extract_label_map(df: pd.DataFrame) -> dict[str, str]:
    if df.empty:
        return {}
    label_map: dict[str, str] = {}
    first_row = df.iloc[0].to_dict()
    for col, value in first_row.items():
        if pd.isna(value):
            continue
        key = str(col).strip()
        label = str(value).strip()
        if key and label:
            label_map[key] = label
    return label_map


def resolve_column_name(columns: list[str], key: str) -> str | None:
    normalized_columns = {_normalize_key(column): column for column in columns}
    return normalized_columns.get(_normalize_key(key))


def pairwise_mean_diff(df: pd.DataFrame, left_key: str, right_key: str) -> float:
    column_names = [str(col) for col in df.columns]
    left_column = resolve_column_name(column_names, left_key)
    right_column = resolve_column_name(column_names, right_key)
    if left_column is None or right_column is None:
        return float("nan")

    left_series = pd.to_numeric(df[left_column], errors="coerce")
    right_series = pd.to_numeric(df[right_column], errors="coerce")
    # Use only rows where both columns have valid numbers.
    valid_mask = left_series.notna() & right_series.notna()
    if not valid_mask.any():
        return float("nan")

    left_mean = float(left_series[valid_mask].mean())
    right_mean = float(right_series[valid_mask].mean())
    return (right_mean - left_mean) * 1000.0


def label_for_key(label_map: dict[str, str], key: str) -> str:
    normalized_labels = {_normalize_key(k): v for k, v in label_map.items()}
    return normalized_labels.get(_normalize_key(key), key)


def output_name_for_formula(formula: Formula, label_map: dict[str, str]) -> str:
    if formula.output:
        return formula.output
    left_label = label_for_key(label_map, formula.left)
    right_label = label_for_key(label_map, formula.right)
    return f"{left_label}->{right_label} (ms)"


def process_time_analysis(input_path: Path, disable_encode: bool = False) -> int:
    input_path = input_path.expanduser().resolve()
    rules_path = resolve_rules_path()
    output_path = input_path.parent / "process_time_analysis.xlsx"

    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not rules_path.is_file():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    formulas = parse_formula_file(rules_path)
    if disable_encode:
        formulas = [f for f in formulas if not _formula_uses_encode_column(f)]
        if not formulas:
            raise ValueError(
                "No formulas remain after --disable-encode; check time_analysis_rules.yaml"
            )
    df = pd.read_excel(input_path, engine="openpyxl")
    label_map = extract_label_map(df)
    output_names = [output_name_for_formula(f, label_map) for f in formulas]
    if len(output_names) != len(set(output_names)):
        raise ValueError("Duplicate output column names found in rules file")

    row: dict[str, object] = {
        "序号": 1,
        "源文件名": input_path.name,
    }

    for f, output_name in zip(formulas, output_names):
        value = pairwise_mean_diff(df, f.left, f.right)
        if pd.isna(value):
            row[output_name] = pd.NA
        else:
            row[output_name] = value

    ordered_cols = ["序号", "源文件名"] + output_names
    out_df = pd.DataFrame([row]).reindex(columns=ordered_cols)
    out_df = out_df.fillna("-").replace(r"^\s*$", "-", regex=True)
    out_df.to_excel(output_path, index=False, engine="openpyxl")
    print(f"Processed time analysis saved to: {output_path}")
    return 0


if __name__ == "__main__":
    # Usage: python process_time_analysis.py <time_analysis.xlsx> [--disable-encode]
    parser = argparse.ArgumentParser(
        description="Aggregate time_analysis.xlsx into process_time_analysis.xlsx using time_analysis_rules.yaml."
    )
    parser.add_argument("input_path", type=Path, help="Path to time_analysis.xlsx")
    parser.add_argument(
        "--disable-encode",
        action="store_true",
        help="Skip formulas referencing encode trace columns (paired with parse_logs --disable-encode).",
    )
    args = parser.parse_args()
    process_time_analysis(args.input_path, disable_encode=args.disable_encode)
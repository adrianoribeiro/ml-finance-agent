"""Data drift detection using PSI (Population Stability Index)."""
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PSI thresholds
PSI_WARNING = 0.1
PSI_RETRAIN = 0.2


def calculate_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Calculate PSI between two distributions."""
    eps = 1e-4

    # Create bins from reference data
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        bins + 1,
    )

    ref_counts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    cur_counts = np.histogram(current, bins=breakpoints)[0] / len(current)

    # Avoid zeros
    ref_counts = np.clip(ref_counts, eps, None)
    cur_counts = np.clip(cur_counts, eps, None)

    psi = np.sum((cur_counts - ref_counts) * np.log(cur_counts / ref_counts))
    return float(psi)


def detect_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_col: str = "SeriousDlqin2yrs",
) -> dict:
    """Compare reference (training) data against current (production) data."""
    ref = reference_df.drop(columns=[target_col], errors="ignore")
    cur = current_df.drop(columns=[target_col], errors="ignore")

    columns = [c for c in ref.columns if c in cur.columns]
    psi_scores = {}
    drifted = []

    for col in columns:
        psi = calculate_psi(ref[col].values, cur[col].values)
        psi_scores[col] = round(psi, 4)
        if psi > PSI_WARNING:
            drifted.append({"column": col, "psi": round(psi, 4)})

    share_drifted = len(drifted) / len(columns) if columns else 0

    status = "OK"
    if share_drifted > PSI_RETRAIN:
        status = "RETRAIN"
    elif share_drifted > PSI_WARNING:
        status = "WARNING"

    output = {
        "status": status,
        "share_drifted_columns": round(share_drifted, 4),
        "number_drifted_columns": len(drifted),
        "total_columns": len(columns),
        "drifted_columns": drifted,
        "psi_per_column": psi_scores,
        "thresholds": {"warning": PSI_WARNING, "retrain": PSI_RETRAIN},
    }

    logger.info(f"Drift status: {status} ({len(drifted)}/{len(columns)} drifted)")
    return output


def generate_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str = "reports/drift_report.html",
) -> str:
    """Generate HTML drift report using Evidently."""
    from evidently import Report
    from evidently.presets import DataDriftPreset

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ref = reference_df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")
    cur = current_df.drop(columns=["SeriousDlqin2yrs"], errors="ignore")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=cur)

    # Evidently 0.7 save
    with open(output_path, "w") as f:
        f.write(report.render())

    logger.info(f"Drift report saved to {output_path}")
    return output_path

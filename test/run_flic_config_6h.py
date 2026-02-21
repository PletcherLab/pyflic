from __future__ import annotations

import argparse
from pathlib import Path

from pyflic import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load a FLIC experiment from a project directory, restrict to the first N hours, "
            "and write feeding summary, experiment summary, and QC reports. "
            "Reads flic_config.yaml and data/ from the project directory; "
            "outputs go to project_dir/analysis/ and project_dir/qc/."
        )
    )
    parser.add_argument(
        "project_dir",
        help="Path to the project directory containing flic_config.yaml.",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=6.0,
        help="Hours of data to include from start (default: %(default)s).",
    )
    parser.add_argument(
        "--qc-data-breaks-multiplier",
        type=float,
        default=4.0,
        help="Multiplier for data break detection threshold (default: %(default)s).",
    )
    parser.add_argument(
        "--qc-bleeding-cutoff",
        type=float,
        default=50.0,
        help="Cutoff for bleeding check (default: %(default)s).",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel loading/calculation for DFMs.",
    )
    args = parser.parse_args()

    minutes = float(args.hours) * 60.0
    exp = Experiment.load(
        args.project_dir,
        range_minutes=(0.0, minutes),
        parallel=not args.no_parallel,
        executor="threads",
    )

    qc_out = exp.write_qc_reports(
        data_breaks_multiplier=float(args.qc_data_breaks_multiplier),
        bleeding_cutoff=float(args.qc_bleeding_cutoff),
    )
    print(f"Wrote QC reports to {qc_out}")

    summary_path = exp.write_summary(
        include_qc=True,
        qc_data_breaks_multiplier=float(args.qc_data_breaks_multiplier),
        qc_bleeding_cutoff=float(args.qc_bleeding_cutoff),
    )
    print(f"Wrote experiment summary to {summary_path}")

    design_df = exp.design.design_table().sort_values(["Treatment", "DFM", "Chamber"])
    summary_df = exp.design.feeding_summary()
    if not summary_df.empty and all(c in summary_df.columns for c in ("Treatment", "DFM", "Chamber")):
        summary_df = summary_df.sort_values(["Treatment", "DFM", "Chamber"])

    out_path = exp.analysis_dir / f"feeding_summary_first_{args.hours:.0f}h.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("FLIC experiment test run\n")
        f.write("=======================\n\n")
        f.write(f"Project dir: {exp.project_dir}\n")
        f.write(f"Data dir: {exp.data_dir}\n")
        f.write(f"Range minutes: (0, {minutes})\n")
        f.write(f"Loaded DFMs: {sorted(exp.dfms.keys())}\n")
        f.write(f"Treatments: {sorted(exp.design.treatments.keys())}\n\n")

        f.write("Experiment design (DFM/Chamber -> Treatment)\n")
        f.write("--------------------------------------------\n")
        if design_df.empty:
            f.write("(empty)\n\n")
        else:
            f.write(design_df.to_string(index=False))
            f.write("\n\n")

        f.write("Feeding summary\n")
        f.write("--------------\n")
        if summary_df.empty:
            f.write("(empty)\n")
        else:
            f.write(summary_df.to_string(index=False))
            f.write("\n")

    print(f"Wrote feeding summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

from pyflic import Experiment


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Load a FLIC experiment from a project directory and write outputs. "
            "The directory must contain flic_config.yaml; data is read from "
            "<project_dir>/data and outputs are written to <project_dir>/qc "
            "and <project_dir>/analysis."
        )
    )
    parser.add_argument("project_dir", help="Path to the project directory.")
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel DFM loading.",
    )
    parser.add_argument(
        "--executor",
        choices=["threads", "processes"],
        default="threads",
        help="Parallel executor type (default: %(default)s).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=0,
        help="Max parallel workers (0 means default).",
    )
    args = parser.parse_args()

    exp = Experiment.load(
        args.project_dir,
        parallel=not args.no_parallel,
        executor=args.executor,
        max_workers=None if args.max_workers == 0 else args.max_workers,
    )

    design = exp.design.design_table()
    summary = exp.design.feeding_summary()

    # Write design table and feeding summary to analysis dir.
    analysis = exp.analysis_dir
    analysis.mkdir(parents=True, exist_ok=True)
    design.to_csv(analysis / "experiment_design.csv", index=False)
    summary.to_csv(analysis / "feeding_summary.csv", index=False)
    print(f"Wrote experiment_design.csv and feeding_summary.csv to {analysis}")

    # Write QC reports to qc dir.
    qc_out = exp.write_qc_reports()
    print(f"Wrote QC reports to {qc_out}")

    # Write summary text to analysis dir.
    summary_path = exp.write_summary()
    print(f"Wrote summary to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

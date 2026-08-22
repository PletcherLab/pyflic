from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

from .experiment import Experiment


@dataclass(slots=True)
class SingleWellExperiment(Experiment):
    """
    A single-well specialisation of :class:`Experiment`.

    All DFMs in the config must use ``chamber_size=1`` (12 independent wells).
    Enforced at load time; raises :class:`ValueError` otherwise.
    """

    @classmethod
    def load(
        cls,
        experiment_dir: str | Path,
        *,
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> SingleWellExperiment:
        """Load a single-well experiment, validating that every DFM uses ``chamber_size=1``."""
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            experiment_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )
        bad = [
            dfm_id
            for dfm_id, dfm in base.dfms.items()
            if dfm.params.chamber_size != 1
        ]
        if bad:
            raise ValueError(
                f"SingleWellExperiment requires chamber_size=1 for every DFM, "
                f"but DFM(s) {sorted(bad)} have chamber_size != 1.  "
                f"Set chamber_size: 1 in flic_config.yaml."
            )
        return cls(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)})

    # ------------------------------------------------------------------
    # Feeding summary plot
    # ------------------------------------------------------------------

    def _feeding_plot_metrics(self) -> list[tuple[str, str]]:
        return [
            ("Licks", "Licks"),
            ("Events", "Events"),
            ("MeanDuration", "Mean Duration (s)"),
            ("MedDuration", "Median Duration (s)"),
            ("MeanTimeBtw", "Mean Time Btw (s)"),
            ("MedTimeBtw", "Median Time Btw (s)"),
            ("MeanInt", "Mean Interval (s)"),
            ("MedianInt", "Median Interval (s)"),
        ]

    def _feeding_summary_default_ncols(self) -> int:
        return 3

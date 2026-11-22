from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import pandas as pd

from .dfm import DFM
from .treatment import Treatment, TreatmentChamber


@dataclass(slots=True)
class ExperimentDesign:
    """
    A mapping of treatment-name -> Treatment, representing an experiment's chamber assignments.
    """

    dfms: dict[int, DFM] = field(default_factory=dict)
    treatments: dict[str, Treatment] = field(default_factory=dict)
    experiment_type: str | None = None  # e.g. "hedonic", "progressive_ratio", "two_well", "single_well"

    def add_dfm(self, dfm: DFM, *, overwrite: bool = False) -> None:
        dfm_id = int(dfm.id)
        if (dfm_id in self.dfms) and not overwrite:
            raise ValueError(f"DFM {dfm_id} already exists in design (use overwrite=True to replace).")
        self.dfms[dfm_id] = dfm

    def add_treatment(self, treatment: Treatment, *, overwrite: bool = False) -> None:
        if (treatment.name in self.treatments) and not overwrite:
            raise ValueError(f"Treatment '{treatment.name}' already exists (use overwrite=True to replace).")
        self.treatments[treatment.name] = treatment

    def remove_treatment(self, name: str) -> None:
        self.treatments.pop(name)

    def get(self, name: str) -> Treatment:
        return self.treatments[name]

    def __iter__(self):
        return iter(self.treatments.values())

    def design_table(self) -> pd.DataFrame:
        """
        Return a table like the R `expDesign` data frame: columns DFM, Chamber, Treatment.
        """

        rows: list[dict[str, object]] = []
        for name, trt in self.treatments.items():
            for tc in trt.chambers:
                rows.append({"DFM": tc.dfm_id, "Chamber": tc.chamber_index, "Treatment": name})
        return pd.DataFrame(rows)

    def with_dfms(self, dfms: dict[int, DFM]) -> "ExperimentDesign":
        """This design over *dfms*, the DFMs recomputed with other
        Parameters: the same treatments, each chamber now the chamber of the
        same index in the DFM of the same id.  ``feeding_summary`` reads the
        treatments' chambers, so replacing ``dfms`` alone left a serial
        experiment summarising the old DFMs.  A DFM not in *dfms* is kept."""
        from dataclasses import replace

        treatments: dict[str, Treatment] = {}
        for name, treatment in self.treatments.items():
            rebound = Treatment(name=name)
            for tc in treatment.chambers:
                rebound.add_chamber(dfms.get(tc.dfm_id, tc.dfm), tc.chamber_index)
            treatments[name] = rebound
        return replace(self, dfms={**self.dfms, **dfms}, treatments=treatments)

    def treatment_for(self, dfm_id: int, chamber_index: int) -> str | None:
        """Return the treatment name for a given DFM/chamber, or None if unassigned."""
        for name, trt in self.treatments.items():
            for tc in trt.chambers:
                if tc.dfm_id == int(dfm_id) and tc.chamber_index == int(chamber_index):
                    return name
        return None

    def feeding_summary(
        self, *, range_minutes: Sequence[float] = (0, 0), transform_licks: bool = True
    ) -> pd.DataFrame:
        """
        Concatenate per-treatment feeding summaries into one DataFrame.
        """

        frames: list[pd.DataFrame] = []
        for trt in self.treatments.values():
            df = trt.feeding_summary(range_minutes=range_minutes, transform_licks=transform_licks)
            if not df.empty:
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


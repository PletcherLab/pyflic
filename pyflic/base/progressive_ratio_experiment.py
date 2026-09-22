"""The Progressive Ratio experiment: paired/yoked Chamber Groups, data-derived
Training/Test Facets, and the within-group difference (ADR-0013).

Vocabulary is fixed in ``CONTEXT.md`` under *Progressive Ratio*:

* a **Chamber Group** is a fixed pair of adjacent chambers (1+2, 3+4, 5+6)
  that shares one light circuit, one Treatment and one training end;
* the **Paired** chamber's feeding at the **Sucrose Well** (always well A)
  drives the light; the **Yoked** chamber is lit at the same moments;
* **Training** ends, per group, at the last minute the paired chamber's
  sucrose well is flagged in training by the firmware (raw sample > 40000);
  the **Test** phase is everything after.

Everything here is derived from the ordinary two-well machinery plus one config
key per DFM, ``paired_chambers``.  The light-on state of a chamber is the OR of
its two wells' ``OptoCol1`` bits (the data sets all four bits of a group
together).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd

from . import windowing
from .dfm import DFM
from .experiment import Experiment
from .experiment_types.progressive_ratio import (
    CHAMBER_GROUPS,
    PAIRED_KEY,
    ROLE_PAIRED,
    ROLE_YOKED,
    group_of,
    parse_paired_chambers,
    partner_of,
)
from .two_well_experiment import TwoWellExperiment
from .utils import range_bounds, range_is_specified

TRAINING_LABEL = "Training"
TEST_LABEL = "Test"
FACET_LABELS = (TRAINING_LABEL, TEST_LABEL)

#: Per-chamber metrics the Paired-Yoked Difference table carries, as ``d<name>``.
DIFF_METRICS: tuple[str, ...] = (
    "LicksA", "LicksB", "EventsA", "EventsB", "PI", "EventPI",
    "MedDurationA", "MedDurationB",
)

#: Two wells clearing within this many minutes of each other count as together.
_FLAG_TOLERANCE_MIN = 0.1


@dataclass(frozen=True, slots=True)
class ChamberRole:
    dfm_id: int
    chamber: int
    group: int
    role: str          # ROLE_PAIRED | ROLE_YOKED
    partner: int


@dataclass(frozen=True, slots=True)
class GroupTraining:
    """One Chamber Group's training verdict."""

    dfm_id: int
    group: int
    paired_chamber: int
    yoked_chamber: int
    sucrose_well: int
    training_end: float | None     # minutes; None = never completed
    notes: tuple[str, ...] = ()    # per-well flag disagreements (QC, never fatal)

    @property
    def complete(self) -> bool:
        return self.training_end is not None


@dataclass(slots=True)
class ProgressiveRatioExperiment(TwoWellExperiment):
    """A Progressive Ratio specialisation of :class:`TwoWellExperiment`."""

    _pr_paired: dict | None = None
    _pr_training: dict | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        experiment_dir: str | Path,
        *,
        range_minutes: Sequence[float] = (0, 0),
        parallel: bool = True,
        max_workers: int | None = None,
        executor: Literal["threads", "processes"] = "threads",
    ) -> ProgressiveRatioExperiment:
        from .yaml_config import load_experiment_yaml

        base = load_experiment_yaml(
            experiment_dir,
            range_minutes=range_minutes,
            parallel=parallel,
            max_workers=max_workers,
            executor=executor,
        )
        if isinstance(base, cls):
            return base
        return cls(**{f.name: getattr(base, f.name)
                      for f in dataclasses.fields(Experiment)})

    # ------------------------------------------------------------------
    # Roles: paired / yoked within Chamber Groups
    # ------------------------------------------------------------------

    def paired_chambers_by_dfm(self) -> dict[int, list[int]]:
        """``{dfm_id: [paired chamber per group]}`` from the config's
        ``paired_chambers`` entries (validated at load time)."""
        if self._pr_paired is not None:
            return self._pr_paired
        nodes = (self.config or {}).get("dfms", (self.config or {}).get("DFMs")) or []
        if isinstance(nodes, dict):
            nodes = [{"id": int(k), **dict(v)} for k, v in nodes.items()]
        out: dict[int, list[int]] = {}
        for node in nodes:
            dfm_id = int(node.get("id", node.get("ID")))
            if dfm_id not in self.dfms:
                continue
            paired, problems = parse_paired_chambers(node.get(PAIRED_KEY))
            if problems:
                raise ValueError(
                    f"DFM {dfm_id}: " + "; ".join(problems))
            out[dfm_id] = paired
        missing = sorted(set(self.dfms) - set(out))
        if missing:
            raise ValueError(
                f"DFM(s) {missing} have no '{PAIRED_KEY}' entry in the config.")
        self._pr_paired = out
        return out

    def role_of(self, dfm_id: int, chamber: int) -> ChamberRole:
        dfm_id, chamber = int(dfm_id), int(chamber)
        paired = self.paired_chambers_by_dfm()[dfm_id]
        role = ROLE_PAIRED if chamber in paired else ROLE_YOKED
        return ChamberRole(dfm_id=dfm_id, chamber=chamber, group=group_of(chamber),
                           role=role, partner=partner_of(chamber))

    def roles_table(self) -> pd.DataFrame:
        """``DFM, Chamber, Group, Role, Partner`` for every chamber of every
        loaded DFM, assigned to a Treatment or not."""
        rows = []
        for dfm_id in sorted(self.dfms):
            for chamber in range(1, 7):
                r = self.role_of(dfm_id, chamber)
                rows.append({"DFM": dfm_id, "Chamber": chamber, "Group": r.group,
                             "Role": r.role, "Partner": r.partner})
        return pd.DataFrame(rows, columns=["DFM", "Chamber", "Group", "Role", "Partner"])

    def _sucrose_well(self, dfm: DFM, chamber: int) -> int:
        return int(dfm.chambers[int(chamber) - 1].well_a)

    def _chamber_wells(self, dfm: DFM, chamber: int) -> tuple[int, int]:
        ch = dfm.chambers[int(chamber) - 1]
        return int(ch.left_well), int(ch.right_well)

    # ------------------------------------------------------------------
    # Training end, per Chamber Group
    # ------------------------------------------------------------------

    def group_training(self, dfm_id: int, group: int) -> GroupTraining:
        """The group's training end, read from the paired chamber's sucrose
        well, plus per-well notes where the other wells' flags disagree."""
        dfm_id, group = int(dfm_id), int(group)
        cache = self._pr_training if self._pr_training is not None else {}
        key = (dfm_id, group)
        if key in cache:
            return cache[key]
        dfm = self.dfms[dfm_id]
        chambers = CHAMBER_GROUPS[group]
        paired = next(c for c in chambers
                      if c in self.paired_chambers_by_dfm()[dfm_id])
        yoked = partner_of(paired)
        sucrose = self._sucrose_well(dfm, paired)
        end = dfm.training_end_minutes(sucrose)

        notes: list[str] = []
        any_flag = False
        ## Wells in the same state are reported together: with the firmware
        ## seen so far every non-sucrose well stays flagged, and three
        ## identical lines per group would bury the one that matters.
        stayed: list[str] = []
        never: list[str] = []
        for chamber in chambers:
            role = ROLE_PAIRED if chamber == paired else ROLE_YOKED
            for well in self._chamber_wells(dfm, chamber):
                state = dfm.training_flag_state(well)
                any_flag = any_flag or state != "never_flagged"
                if well == sucrose:
                    continue
                label = (f"W{well} ({role} ch{chamber} "
                         f"{'A' if well == self._sucrose_well(dfm, chamber) else 'B'})")
                if state == "never_cleared":
                    stayed.append(label)
                elif state == "never_flagged":
                    never.append(label)
                else:
                    t = dfm.training_end_minutes(well)
                    if end is None:
                        notes.append(f"{label} cleared at {t:.1f} min while the "
                                     f"paired sucrose well W{sucrose} never cleared")
                    elif abs(float(t) - float(end)) > _FLAG_TOLERANCE_MIN:
                        notes.append(f"{label} cleared at {t:.1f} min, paired "
                                     f"sucrose well W{sucrose} at {end:.1f} min")
        if stayed:
            notes.append(f"{', '.join(stayed)} stayed flagged in training to the "
                         f"end of the recording")
        if never and any_flag:
            notes.append(f"{', '.join(never)} never carried the training flag")
        if not any_flag:
            ## No flag on any well of the group: not a disagreement between
            ## wells but an absence of training data (a v2 file, or a run that
            ## never entered training).  One note, not four.
            notes = [f"no training flag on any well of group {group}"]
        elif end is None:
            state = dfm.training_flag_state(sucrose)
            notes.insert(0, f"paired sucrose well W{sucrose} "
                            f"{'stayed flagged to the end of the recording' if state == 'never_cleared' else 'never carried the training flag'}"
                            f" — training never completed")
        result = GroupTraining(dfm_id=dfm_id, group=group, paired_chamber=paired,
                               yoked_chamber=yoked, sucrose_well=sucrose,
                               training_end=end, notes=tuple(notes))
        cache[key] = result
        self._pr_training = cache
        return result

    def training_table(self) -> pd.DataFrame:
        """One row per Chamber Group: ``DFM, Group, PairedChamber, YokedChamber,
        SucroseWell, TrainingEndMin, TrainingComplete, Notes``."""
        rows = []
        for dfm_id in sorted(self.dfms):
            for group in CHAMBER_GROUPS:
                gt = self.group_training(dfm_id, group)
                rows.append({
                    "DFM": dfm_id, "Group": group,
                    "PairedChamber": gt.paired_chamber,
                    "YokedChamber": gt.yoked_chamber,
                    "SucroseWell": gt.sucrose_well,
                    "TrainingEndMin": (np.nan if gt.training_end is None
                                       else float(gt.training_end)),
                    "TrainingComplete": bool(gt.complete),
                    "Notes": "; ".join(gt.notes),
                })
        return pd.DataFrame(rows)

    def training_warnings(self) -> list[str]:
        """Human-readable per-well flag disagreements, for the summary and QC."""
        out: list[str] = []
        for dfm_id in sorted(self.dfms):
            for group in CHAMBER_GROUPS:
                gt = self.group_training(dfm_id, group)
                for note in gt.notes:
                    out.append(f"DFM {dfm_id} group {group}: {note}")
        return out

    def _recording_end(self, dfm: DFM) -> float:
        return float(pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").max())

    # ------------------------------------------------------------------
    # Light on, per chamber
    # ------------------------------------------------------------------

    def _chamber_light(self, dfm: DFM, chamber: int) -> pd.Series:
        """Boolean per-sample light state for *chamber*: either well's bit."""
        w1, w2 = self._chamber_wells(dfm, chamber)
        lights = dfm.lights_df
        if lights is None or f"W{w1}" not in lights.columns:
            return pd.Series(False, index=lights.index if lights is not None else [])
        return lights[f"W{w1}"].astype(bool) | lights[f"W{w2}"].astype(bool)

    def _light_on_seconds(self, dfm: DFM, chamber: int,
                          range_minutes: Sequence[float]) -> float:
        on = self._chamber_light(dfm, chamber)
        if range_is_specified(range_minutes):
            a, b = range_bounds(range_minutes)
            mins = dfm.lights_df["Minutes"].to_numpy(dtype=float)
            on = on[(mins > a) & (mins <= b)]
        return float(on.sum()) / float(dfm.params.samples_per_second)

    # ------------------------------------------------------------------
    # Feeding summary: standard two-well columns + Group/Role/Training/Light
    # ------------------------------------------------------------------

    def _augment_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Append ``Group, Role, TrainingMinutes, TrainingComplete, LightOn_sec``
        to a per-chamber frame carrying ``DFM, Chamber, StartMin, EndMin``."""
        if df is None or df.empty or "Role" in df.columns:
            return df
        df = df.copy()
        groups, roles, tmins, tdone, light = [], [], [], [], []
        for _, row in df.iterrows():
            dfm_id, chamber = int(row["DFM"]), int(row["Chamber"])
            r = self.role_of(dfm_id, chamber)
            gt = self.group_training(dfm_id, r.group)
            groups.append(r.group)
            roles.append(r.role)
            ## Training end is a property of the Chamber Group, not of the
            ## paired fly alone: it is the moment the group's light stopped
            ## being purely closed-loop, and the yoked fly lived through the
            ## same moment.  So both rows carry it — the yoked cell used to be
            ## NA, which read as missing data in every table and plot that
            ## groups by it.  A group that never finished has no such moment,
            ## and both its rows stay NA.
            tmins.append(float(gt.training_end) if gt.complete else np.nan)
            tdone.append(bool(gt.complete))
            rng = (float(row.get("StartMin", 0.0)), float(row.get("EndMin", 0.0)))
            light.append(self._light_on_seconds(self.dfms[dfm_id], chamber, rng))
        pos = list(df.columns).index("Chamber") + 1
        df.insert(pos, "Group", groups)
        df.insert(pos + 1, "Role", roles)
        df["TrainingMinutes"] = tmins
        df["TrainingComplete"] = tdone
        df["LightOn_sec"] = light
        return df

    def feeding_summary(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        if transform_licks is None:
            transform_licks = self.transform_licks
        base = Experiment.feeding_summary(
            self, range_minutes=range_minutes, transform_licks=transform_licks)
        if base is None or base.empty or "Role" in base.columns:
            return base
        out = self._augment_rows(base)
        key = (float(range_minutes[0]), float(range_minutes[1])), bool(transform_licks)
        self._feeding_summary_cache[key] = out
        return out

    def _chamber_rows(self, dfm_id: int, chambers: Sequence[int],
                      range_minutes: Sequence[float],
                      transform_licks: bool) -> pd.DataFrame:
        """Treatment-labelled summary rows for *chambers* of one DFM over one
        window — the per-group building block of the faceted summary."""
        dfm = self.dfms[int(dfm_id)]
        wanted = {int(c) for c in chambers
                  if self.design.treatment_for(dfm_id, c) is not None}
        if not wanted:
            return pd.DataFrame()
        summ = dfm.feeding_summary(range_minutes=range_minutes,
                                   transform_licks=transform_licks)
        summ = summ[summ["Chamber"].astype(int).isin(wanted)].copy()
        if summ.empty:
            return summ
        summ.insert(0, "Treatment",
                    [self.design.treatment_for(dfm_id, int(c)) for c in summ["Chamber"]])
        return self._augment_rows(self._append_factor_columns(summ))

    # ------------------------------------------------------------------
    # Facets: Training / Test, split at each group's own training end
    # ------------------------------------------------------------------

    def resolved_facet_cutoffs(self):
        return None

    def facet_windows(self):
        """Empty: this type's windows are per Chamber Group, see
        :meth:`group_windows`.  Consumers that need "is this faceted" should
        use :meth:`facet_labels`."""
        return []

    def facet_labels(self):
        return list(FACET_LABELS)

    def group_windows(self, dfm_id: int, group: int) -> list[tuple[str, tuple]]:
        """``[(label, (start, end)), ...]`` for one Chamber Group.  A group
        that never completed training has only a Training window, spanning
        the whole recording."""
        gt = self.group_training(dfm_id, group)
        if not gt.complete:
            return [(TRAINING_LABEL, (0.0, float("inf")))]
        ## Rounded to the hundredth of a minute so the FacetRange cell and the
        ## StartMin/EndMin columns read as a time, not a float dump; both sides
        ## of the split use the same rounded value, so the partition is exact.
        end = round(float(gt.training_end), 2)
        return [(TRAINING_LABEL, (0.0, end)), (TEST_LABEL, (end, float("inf")))]

    def feeding_summary_facet(
        self,
        *,
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        if transform_licks is None:
            transform_licks = self.transform_licks
        frames: list[pd.DataFrame] = []
        for dfm_id in sorted(self.dfms):
            for group, chambers in CHAMBER_GROUPS.items():
                for label, window in self.group_windows(dfm_id, group):
                    df = self._chamber_rows(
                        dfm_id, chambers,
                        range_minutes=windowing.as_range_minutes(window),
                        transform_licks=bool(transform_licks))
                    if df is None or df.empty:
                        continue
                    df = df.copy()
                    df.insert(0, "FacetRange", windowing.format_range(window))
                    df.insert(0, "Facet", label)
                    frames.append(df)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True)
        order = {TRAINING_LABEL: 0, TEST_LABEL: 1}
        out["_o"] = out["Facet"].map(order)
        out = out.sort_values(["_o", "DFM", "Chamber"], kind="stable").drop(columns="_o")
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Paired-Yoked Difference
    # ------------------------------------------------------------------

    def paired_yoked_diff(
        self,
        *,
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """One row per Chamber Group per Facet: paired minus yoked for each of
        :data:`DIFF_METRICS` (as ``d<metric>``).  A group missing either
        chamber contributes no row."""
        facet = self.feeding_summary_facet(transform_licks=transform_licks)
        cols = ["Treatment", *(self.design_factors or []), "DFM", "Group", "Facet",
                "FacetRange", "PairedChamber", "YokedChamber", "StartMin", "EndMin",
                "TrainingMinutes", "TrainingComplete", "LightOn_sec",
                *(f"d{m}" for m in DIFF_METRICS)]
        if facet is None or facet.empty:
            return pd.DataFrame(columns=cols)
        rows = []
        keys = ["DFM", "Group", "Facet"]
        for (dfm_id, group, label), sub in facet.groupby(keys, sort=False):
            paired = sub[sub["Role"] == ROLE_PAIRED]
            yoked = sub[sub["Role"] == ROLE_YOKED]
            if len(paired) != 1 or len(yoked) != 1:
                continue
            p, y = paired.iloc[0], yoked.iloc[0]
            row = {"Treatment": p["Treatment"]}
            for f in (self.design_factors or []):
                row[f] = p.get(f, "")
            row.update({
                "DFM": int(dfm_id), "Group": int(group), "Facet": label,
                "FacetRange": p["FacetRange"],
                "PairedChamber": int(p["Chamber"]), "YokedChamber": int(y["Chamber"]),
                "StartMin": float(p["StartMin"]), "EndMin": float(p["EndMin"]),
                "TrainingMinutes": p["TrainingMinutes"],
                "TrainingComplete": bool(p["TrainingComplete"]),
                "LightOn_sec": float(p["LightOn_sec"]),
            })
            for m in DIFF_METRICS:
                pv, yv = p.get(m, np.nan), y.get(m, np.nan)
                row[f"d{m}"] = (float(pv) - float(yv)
                                if pd.notna(pv) and pd.notna(yv) else np.nan)
            rows.append(row)
        return pd.DataFrame(rows, columns=cols)

    def write_paired_yoked_diff(
        self,
        path: str | Path | None = None,
        *,
        transform_licks: bool | None = None,
    ) -> Path:
        if path is None:
            if self.analysis_dir is None:
                raise ValueError("path must be provided when no experiment_dir is set.")
            path = self.analysis_dir / "paired_yoked_diff.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.paired_yoked_diff(transform_licks=transform_licks).to_csv(
            out, index=False, na_rep="NA")
        return out

    # ------------------------------------------------------------------
    # The standard per-treatment plots, made role-aware
    # ------------------------------------------------------------------
    #
    # A Treatment in this type names *both* flies of a Chamber Group, so the
    # inherited plots — which group by Treatment alone — draw one cloud of
    # points per treatment holding paired and yoked flies together.  That
    # cloud has no referent: the yoked fly's PI is not a second measurement
    # of the paired fly's preference, it is the control the paired fly is
    # measured *against*.  Pooling them averages an effect with its own
    # control and lands halfway to nothing.
    #
    # Two fixes, one per plot shape.  The time courses and the multi-metric
    # feeding summary split their groups by Role, so a treatment becomes two
    # series — the honest version of the same picture.  The dot plot, where
    # one point is one observation, goes further and plots the within-group
    # difference itself: paired minus yoked, one point per Chamber Group,
    # which is the unit CONTEXT.md fixes for this type ("a difference is
    # always taken *within* a Chamber Group, never between group means").

    def _append_role_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ``Group`` and ``Role`` to any frame carrying ``DFM`` and
        ``Chamber``.

        The cheap half of :meth:`_augment_rows`: roles are a config fact, so
        this costs a dict lookup per row, while the training end and the
        light-on seconds mean reading the signal.  Binned and moving-window
        tables have a row per chamber *per bin* and need only the role.
        """
        if df is None or df.empty or "Role" in df.columns:
            return df
        if "DFM" not in df.columns or "Chamber" not in df.columns:
            return df
        df = df.copy()
        keys = list(zip(df["DFM"].astype(int), df["Chamber"].astype(int)))
        roles = {k: self.role_of(*k) for k in set(keys)}
        pos = list(df.columns).index("Chamber") + 1
        df.insert(pos, "Group", [roles[k].group for k in keys])
        df.insert(pos + 1, "Role", [roles[k].role for k in keys])
        return df

    def _assemble_treatment_table(
        self, summary_by_dfm: dict[int, pd.DataFrame]
    ) -> pd.DataFrame:
        """The inherited treatment table, plus each row's Chamber Group and
        Role — so every plot built on it can keep the two flies apart."""
        return self._append_role_columns(
            Experiment._assemble_treatment_table(self, summary_by_dfm))

    def _resolve_group_col(self, df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
        """The inherited grouping, with the Role appended to every label.

        ``w1118`` becomes ``w1118 · paired`` and ``w1118 · yoked``: one
        series per fly role, never one series holding both.  A frame with no
        ``Role`` column (anything not per-chamber) is grouped as before.
        """
        df, group_col = Experiment._resolve_group_col(self, df)
        if "Role" not in df.columns:
            return df, group_col
        df = df.copy()
        df["_RoleGroup"] = (df[group_col].astype(str) + " · "
                            + df["Role"].astype(str))
        return df, "_RoleGroup"

    def paired_yoked_delta(
        self,
        *,
        metric: str = "PI",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> pd.DataFrame:
        """One row per Chamber Group: ``Delta`` = paired *metric* − yoked.

        The general form of :meth:`paired_yoked_diff`, which carries only the
        fixed :data:`DIFF_METRICS` as stored columns.  Here any metric the
        summary can express — including the A/B combinations behind
        *two_well_mode* — is resolved per chamber first and differenced
        after, so the metric catalogue the UI offers works unchanged.

        With no *range_minutes*, rows come from the per-group Training / Test
        Facets and a ``Facet`` column says which.  Given an explicit window,
        that window is used for every group and ``Facet`` reads ``"Custom"`` —
        the Facets are per Chamber Group, so one shared window is a different
        question, not a filter on the same one.

        Groups missing either chamber (excluded, or never assigned) contribute
        no row, exactly as in :meth:`paired_yoked_diff`.
        """
        factors = list(self.design_factors or [])
        cols = ["Treatment", *factors, "DFM", "Group", "Facet",
                "PairedChamber", "YokedChamber", "Paired", "Yoked", "Delta"]
        if range_is_specified(range_minutes):
            per_chamber = self.feeding_summary(
                range_minutes=range_minutes, transform_licks=transform_licks)
            if per_chamber is not None and not per_chamber.empty:
                per_chamber = per_chamber.copy()
                per_chamber["Facet"] = "Custom"
        else:
            per_chamber = self.feeding_summary_facet(
                transform_licks=transform_licks)
        if per_chamber is None or per_chamber.empty:
            return pd.DataFrame(columns=cols)

        per_chamber = per_chamber.copy()
        per_chamber["_Value"] = self._metric_series_from_binned_rows(
            per_chamber, metric=metric, two_well_mode=two_well_mode)

        rows: list[dict] = []
        for (dfm_id, group, label), sub in per_chamber.groupby(
                ["DFM", "Group", "Facet"], sort=False):
            paired = sub[sub["Role"] == ROLE_PAIRED]
            yoked = sub[sub["Role"] == ROLE_YOKED]
            if len(paired) != 1 or len(yoked) != 1:
                continue
            p, y = paired.iloc[0], yoked.iloc[0]
            pv, yv = p["_Value"], y["_Value"]
            row = {"Treatment": p["Treatment"]}
            for f in factors:
                row[f] = p.get(f, "")
            row.update({
                "DFM": int(dfm_id), "Group": int(group), "Facet": label,
                "PairedChamber": int(p["Chamber"]),
                "YokedChamber": int(y["Chamber"]),
                "Paired": float(pv) if pd.notna(pv) else np.nan,
                "Yoked": float(yv) if pd.notna(yv) else np.nan,
                "Delta": (float(pv) - float(yv)
                          if pd.notna(pv) and pd.notna(yv) else np.nan),
            })
            rows.append(row)
        out = pd.DataFrame(rows, columns=cols)
        return out.dropna(subset=["Delta"]).reset_index(drop=True)

    def plot_dot_metric_by_treatment(
        self,
        *,
        metric: str = "Licks",
        two_well_mode: Literal["total", "mean_ab", "A", "B"] = "total",
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
    ) -> Any:
        """Jitter + mean ± SE of the **within-group difference** in *metric*.

        One point is one Chamber Group's paired-minus-yoked value, not one
        fly: the comparison this design licenses.  Zero is drawn, because
        zero — not the axis floor — is the null this plot is read against.

        The x axis is the first design factor (or the Treatment when there
        are none) and the panels are the Facets, with any further factors
        folded into the panel label.
        """
        from plotnine import annotate, ggplot, theme_bw

        df = self.paired_yoked_delta(
            metric=metric, two_well_mode=two_well_mode,
            range_minutes=range_minutes, transform_licks=transform_licks)
        if df.empty:
            return (ggplot()
                    + annotate("text", x=0, y=0,
                               label="No complete chamber group to difference")
                    + theme_bw())

        factor_cols = [f for f in (self.design_factors or []) if f in df.columns]
        x_col = factor_cols[0] if factor_cols else "Treatment"
        ## Facet first, so the panels read Training | Test at a glance; any
        ## factor past the x axis joins the panel label rather than the x.
        panel_cols = ["Facet", *factor_cols[1:]]
        df = df.copy()
        panel = df[panel_cols].astype(str).agg(" / ".join, axis=1)
        ## Ordered by first appearance, which is the order the type defines
        ## its Facets in \u2014 Training then Test, not the alphabet's Test first.
        df["_Panel"] = pd.Categorical(
            panel, categories=list(dict.fromkeys(panel)), ordered=True)

        label = f"\u0394{metric} (paired \u2212 yoked)"
        return self.plot_jitter_summary(
            df,
            x_col=x_col,
            y_col="Delta",
            facet_col="_Panel",
            title="One point per chamber group",
            y_label=label,
            hline_at=0.0,
        )

    # ------------------------------------------------------------------
    # Auto-removal: two-well thresholds + require_training_complete
    # ------------------------------------------------------------------

    def auto_remove_chambers(
        self,
        *,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool = True,
        min_untransformed_licks_cutoff: float | None = None,
    ) -> pd.DataFrame:
        base = Experiment.auto_remove_chambers(
            self, range_minutes=range_minutes,
            min_untransformed_licks_cutoff=min_untransformed_licks_cutoff)
        constants = self.global_constants or {}
        max_dur = constants.get("max_med_duration_cutoff")
        max_events = constants.get("max_events_cutoff")
        require_training = _truthy(constants.get("require_training_complete", True))

        fs = self.feeding_summary(range_minutes=range_minutes,
                                  transform_licks=transform_licks)
        already: set[tuple[int, int]] = set()
        if base is not None and not base.empty:
            already = {(int(r["DFM"]), int(r["Chamber"])) for _, r in base.iterrows()}

        removed_rows: list[dict] = []
        to_remove: set[tuple[int, int]] = set()
        for _, row in fs.iterrows():
            key = (int(row["DFM"]), int(row["Chamber"]))
            if key in already or key in to_remove:
                continue
            reasons: list[str] = []
            for col in ("MedDurationA", "MedDurationB"):
                v = row.get(col)
                if max_dur is not None and pd.notna(v) and float(v) > float(max_dur):
                    reasons.append(f"{col}={float(v):.6g} > max_med_duration_cutoff={max_dur}")
            for col in ("EventsA", "EventsB"):
                v = row.get(col)
                if max_events is not None and pd.notna(v) and float(v) > float(max_events):
                    reasons.append(f"{col}={float(v):.6g} > max_events_cutoff={max_events}")
            if require_training and not bool(row.get("TrainingComplete", True)):
                reasons.append(
                    f"training never completed in chamber group {int(row['Group'])} "
                    f"(require_training_complete)")
            if reasons:
                to_remove.add(key)
                removed_rows.append({"DFM": key[0], "Chamber": key[1],
                                     "Treatment": str(row.get("Treatment", "")),
                                     "Reason": "; ".join(reasons)})
                print(f"  [auto_remove_chambers] Removing DFM {key[0]} Chamber {key[1]}"
                      f" ({row.get('Treatment', '')}): {'; '.join(reasons)}", flush=True)
        if to_remove:
            self._remove_chambers_from_design(to_remove)

        extra = pd.DataFrame(removed_rows, columns=["DFM", "Chamber", "Treatment", "Reason"])
        if base is None or base.empty:
            combined = extra
        elif extra.empty:
            combined = base
        else:
            combined = pd.concat([base, extra], ignore_index=True)
        self.filtered_chambers = combined

        lines = ["", "Progressive Ratio filter (additional criteria)", ""]
        lines.append(f"  • max_med_duration_cutoff = {float(max_dur):g}: excluded if "
                     f"MedDurationA or MedDurationB > {float(max_dur):g}"
                     if max_dur is not None else
                     "  • max_med_duration_cutoff: not configured.")
        lines.append(f"  • max_events_cutoff = {float(max_events):g}: excluded if "
                     f"EventsA or EventsB > {float(max_events):g}"
                     if max_events is not None else
                     "  • max_events_cutoff: not configured.")
        lines.append("  • require_training_complete = true: both chambers of a "
                     "chamber group whose paired fly never finished training are excluded"
                     if require_training else
                     "  • require_training_complete = false: incomplete-training "
                     "groups are kept (TrainingComplete = false).")
        self.filter_criteria_summary = self.filter_criteria_summary + "\n" + "\n".join(lines)
        self.write_removed_chambers()
        return combined

    # ------------------------------------------------------------------
    # Cumulative curves since training end
    # ------------------------------------------------------------------

    def cumulative_curve_data(
        self,
        *,
        binsize_min: float = 1.0,
        well: str = "A",
    ) -> pd.DataFrame:
        """Per-chamber cumulative raw licks on *well* since the group's
        training end, at *binsize_min* resolution.

        Columns: ``Treatment, [factors], DFM, Chamber, Group, Role, Minutes,
        CumLicks, LightOn`` where ``Minutes`` is the bin's right edge in
        minutes since training end and ``LightOn`` says whether the group's
        light was on at any sample in the bin.  Groups that never completed
        training contribute nothing.
        """
        if binsize_min <= 0:
            raise ValueError("binsize_min must be positive.")
        frames: list[pd.DataFrame] = []
        for dfm_id in sorted(self.dfms):
            dfm = self.dfms[dfm_id]
            mins_all = dfm.lick_df["Minutes"].to_numpy(dtype=float)
            for group, chambers in CHAMBER_GROUPS.items():
                gt = self.group_training(dfm_id, group)
                if not gt.complete:
                    continue
                end = float(gt.training_end)
                mask = mins_all > end
                if not mask.any():
                    continue
                rel = mins_all[mask] - end
                bins = np.floor(rel / float(binsize_min)).astype(int)
                edges = (bins + 1) * float(binsize_min)
                for chamber in chambers:
                    if self.design.treatment_for(dfm_id, chamber) is None:
                        continue
                    ch = dfm.chambers[chamber - 1]
                    w = int(ch.well_a if well.upper() == "A" else ch.well_b)
                    licks = dfm.lick_df[f"W{w}"].to_numpy(dtype=float)[mask]
                    light = self._chamber_light(dfm, chamber).to_numpy(dtype=bool)[mask]
                    tmp = pd.DataFrame({"bin": bins, "Minutes": edges,
                                        "licks": licks, "light": light})
                    agg = tmp.groupby("bin", sort=True).agg(
                        Minutes=("Minutes", "first"), licks=("licks", "sum"),
                        LightOn=("light", "any")).reset_index(drop=True)
                    agg["CumLicks"] = agg["licks"].cumsum()
                    r = self.role_of(dfm_id, chamber)
                    agg.insert(0, "Role", r.role)
                    agg.insert(0, "Group", group)
                    agg.insert(0, "Chamber", chamber)
                    agg.insert(0, "DFM", dfm_id)
                    agg.insert(0, "Treatment", self.design.treatment_for(dfm_id, chamber))
                    frames.append(agg.drop(columns="licks"))
        cols = ["Treatment", "DFM", "Chamber", "Group", "Role", "Minutes",
                "CumLicks", "LightOn"]
        if not frames:
            return pd.DataFrame(columns=cols)
        out = pd.concat(frames, ignore_index=True)[cols]
        return self._append_factor_columns(out)

    def cumulative_diff_data(self, *, binsize_min: float = 1.0) -> pd.DataFrame:
        """Per-group paired-minus-yoked cumulative sucrose-well licks since
        training end: ``Treatment, [factors], DFM, Group, Minutes,
        DiffCumLicks``.  The data behind the Cumulative Difference Curve and
        the ``timecourse_pr_diff`` Plot Spec."""
        curves = self.cumulative_curve_data(binsize_min=binsize_min, well="A")
        cols = ["Treatment", *(self.design_factors or []), "DFM", "Group",
                "Minutes", "DiffCumLicks"]
        if curves.empty:
            return pd.DataFrame(columns=cols)
        frames = []
        for (dfm_id, group), sub in curves.groupby(["DFM", "Group"], sort=True):
            p = sub[sub["Role"] == ROLE_PAIRED].set_index("Minutes")["CumLicks"]
            y = sub[sub["Role"] == ROLE_YOKED].set_index("Minutes")["CumLicks"]
            if p.empty or y.empty:
                continue
            idx = p.index.intersection(y.index)
            diff = pd.DataFrame({"Minutes": idx,
                                 "DiffCumLicks": p.loc[idx].to_numpy() - y.loc[idx].to_numpy()})
            first = sub.iloc[0]
            diff.insert(0, "Group", int(group))
            diff.insert(0, "DFM", int(dfm_id))
            for f in reversed(self.design_factors or []):
                diff.insert(0, f, first.get(f, ""))
            diff.insert(0, "Treatment", first["Treatment"])
            frames.append(diff)
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)[cols]

    def cumulative_diff_stat(self, *, binsize_min: float = 1.0) -> pd.DataFrame:
        """Mean ± SEM of the paired-minus-yoked curve per Treatment and bin —
        the line the Cumulative Difference Curve draws.  Runs only as far as
        the shortest Chamber Group, so every group is in every averaged point
        and the curve never jumps when one group's recording ends."""
        data = self.cumulative_diff_data(binsize_min=binsize_min)
        cols = ["Treatment", "Minutes", "mean", "sem", "count", "ymin", "ymax"]
        if data.empty:
            return pd.DataFrame(columns=cols)
        common_end = float(data.groupby(["DFM", "Group"])["Minutes"].max().min())
        stat = (data[data["Minutes"] <= common_end]
                .groupby(["Treatment", "Minutes"], sort=True)["DiffCumLicks"]
                .agg(["mean", "sem", "count"]).reset_index())
        stat["sem"] = stat["sem"].fillna(0.0)
        stat["ymin"] = stat["mean"] - stat["sem"]
        stat["ymax"] = stat["mean"] + stat["sem"]
        return stat[cols]

    def write_cumulative_diff(self, path: str | Path | None = None, *,
                              binsize_min: float = 1.0) -> Path:
        """Write ``analysis/pr_cumulative_diff.csv`` — the binned difference
        curve a Project stacks for the pooled ``timecourse_pr_diff`` figure."""
        if path is None:
            if self.analysis_dir is None:
                raise ValueError("path must be provided when no experiment_dir is set.")
            path = self.analysis_dir / "pr_cumulative_diff.csv"
        out = Path(path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        self.cumulative_diff_data(binsize_min=binsize_min).to_csv(
            out, index=False, na_rep="NA")
        return out

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def plot_cumulative_diff(self, *, binsize_min: float = 1.0,
                             base_font_size: float = 10.0,
                             figsize: tuple[float, float] = (8.0, 5.0)):
        """The Cumulative Difference Curve: mean ± SEM per Treatment of the
        paired-minus-yoked cumulative sucrose-well licks against minutes since
        training end, the individual Chamber Group traces faint behind.  The
        mean is drawn only over the range every group covers."""
        import plotnine as p9

        from .experiment import _OKABE_ITO

        data = self.cumulative_diff_data(binsize_min=binsize_min)
        if data.empty:
            return p9.ggplot() + p9.labs(title="Paired − yoked cumulative licks — no data "
                                               "(no chamber group completed training)")
        data = data.copy()
        data["Trace"] = data["DFM"].astype(str) + ":" + data["Group"].astype(str)
        stat = self.cumulative_diff_stat(binsize_min=binsize_min)
        treatments = list(dict.fromkeys(data["Treatment"]))
        palette = {t: _OKABE_ITO[i % len(_OKABE_ITO)] for i, t in enumerate(treatments)}
        well_a = (self.well_names or {}).get("A", "well A")
        g = (p9.ggplot()
             + p9.geom_line(data, p9.aes("Minutes", "DiffCumLicks", group="Trace",
                                         color="Treatment"), alpha=0.25, size=0.4)
             + p9.geom_hline(yintercept=0.0, linetype="dashed", color="#888888", size=0.3)
             + p9.geom_ribbon(stat, p9.aes("Minutes", ymin="ymin", ymax="ymax",
                                           fill="Treatment"), alpha=0.2, color="none")
             + p9.geom_line(stat, p9.aes("Minutes", "mean", color="Treatment"), size=1.0)
             + p9.scale_color_manual(values=palette)
             + p9.scale_fill_manual(values=palette)
             + p9.labs(title="Paired − yoked cumulative licks since training end",
                       x="Time since training end (min)",
                       y=f"Paired − yoked cumulative licks ({well_a})")
             + p9.theme_bw(base_size=base_font_size)
             + p9.theme(figure_size=figsize, legend_position="right"))
        return g

    def plot_cumulative_licks_dfm(self, dfm_id: int, *, binsize_min: float = 1.0,
                                  base_font_size: float = 10.0,
                                  figsize: tuple[float, float] | None = None):
        """QC figure for one DFM: one panel per Chamber Group, paired and
        yoked cumulative sucrose-well licks since the group's training end,
        light-on bins drawn as points."""
        import plotnine as p9

        dfm_id = int(dfm_id)
        curves = self.cumulative_curve_data(binsize_min=binsize_min, well="A")
        curves = curves[curves["DFM"] == dfm_id] if not curves.empty else curves
        if curves.empty:
            return p9.ggplot() + p9.labs(
                title=f"DFM {dfm_id} — no chamber group completed training")
        curves = curves.copy()
        panel = {}
        for group in CHAMBER_GROUPS:
            gt = self.group_training(dfm_id, group)
            trt = self.design.treatment_for(dfm_id, gt.paired_chamber) or \
                self.design.treatment_for(dfm_id, gt.yoked_chamber) or ""
            end = "n/a" if not gt.complete else f"{gt.training_end:.1f} min"
            panel[group] = f"Group {group} ({trt}) — trained {end}"
        curves["Panel"] = pd.Categorical(curves["Group"].map(panel),
                                         categories=[panel[g] for g in CHAMBER_GROUPS
                                                     if panel[g] in set(curves["Group"].map(panel))],
                                         ordered=True)
        curves["Role"] = pd.Categorical(curves["Role"], categories=[ROLE_PAIRED, ROLE_YOKED])
        lit = curves[curves["LightOn"]]
        colors = {ROLE_PAIRED: "#D55E00", ROLE_YOKED: "#0072B2"}
        well_a = (self.well_names or {}).get("A", "well A")
        n = curves["Panel"].nunique()
        if figsize is None:
            figsize = (4.0 * max(n, 1), 3.6)
        g = (p9.ggplot(curves, p9.aes("Minutes", "CumLicks", color="Role"))
             + p9.geom_line(size=0.7)
             + p9.geom_point(data=lit, size=1.4, alpha=0.9)
             + p9.facet_wrap("~ Panel", ncol=3, scales="free_x")
             + p9.scale_color_manual(values=colors)
             + p9.labs(title=f"DFM {dfm_id} — cumulative {well_a} licks since training end "
                             f"(points: light on)",
                       x="Time since training end (min)", y=f"Cumulative licks ({well_a})")
             + p9.theme_bw(base_size=base_font_size)
             + p9.theme(figure_size=figsize, legend_position="bottom"))
        return g

    def write_pr_figures(self, *, binsize_min: float = 1.0, dpi: int = 200) -> dict[str, Path]:
        """Write the headline curve and the per-DFM QC traces into ``analysis/``."""
        if self.analysis_dir is None:
            raise ValueError("experiment_dir must be set to write figures.")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        path = self.analysis_dir / "pr_cumulative_diff.png"
        self.plot_cumulative_diff(binsize_min=binsize_min).save(str(path), dpi=dpi, verbose=False)
        written["pr_cumulative_diff"] = path
        for dfm_id in sorted(self.dfms):
            path = self.analysis_dir / f"pr_cumulative_licks_dfm{dfm_id}.png"
            self.plot_cumulative_licks_dfm(dfm_id, binsize_min=binsize_min).save(
                str(path), dpi=dpi, verbose=False)
            written[f"pr_cumulative_licks_dfm{dfm_id}"] = path
        return written

    # ------------------------------------------------------------------
    # Breaking point — spirit of breaking_point.R on the new model
    # ------------------------------------------------------------------

    def breaking_point_table(self, dfm_id: int, chamber: int) -> pd.DataFrame:
        """Per-light-on-period table for one chamber after its group's training
        end: one row at each moment the group's light switches on, with
        ``Minutes`` (since training end), ``CumLicks`` (sucrose well, since
        training end), ``DeltaMinutes`` and ``DeltaLicks`` since the previous
        onset.  Empty for a group that never completed training.

        The details are provisional (see the decided-direction notes in
        ``docs/progressive-ratio-implementation.md``); what is kept from the R
        port is the shape: ΔLicks per light period, read against time.
        """
        cols = ["Minutes", "CumLicks", "DeltaMinutes", "DeltaLicks"]
        dfm_id, chamber = int(dfm_id), int(chamber)
        dfm = self.dfms[dfm_id]
        gt = self.group_training(dfm_id, group_of(chamber))
        if not gt.complete:
            return pd.DataFrame(columns=cols)
        end = float(gt.training_end)
        mins = dfm.lick_df["Minutes"].to_numpy(dtype=float)
        mask = mins > end
        if not mask.any():
            return pd.DataFrame(columns=cols)
        well_a = self._sucrose_well(dfm, chamber)
        cum = dfm.lick_df[f"W{well_a}"].to_numpy(dtype=float)[mask].cumsum()
        light = self._chamber_light(dfm, chamber).to_numpy(dtype=bool)[mask]
        rel = mins[mask] - end
        onsets = np.flatnonzero(light & ~np.concatenate([[False], light[:-1]]))
        if onsets.size == 0:
            return pd.DataFrame(columns=cols)
        out = pd.DataFrame({"Minutes": rel[onsets], "CumLicks": cum[onsets]})
        out["DeltaMinutes"] = np.concatenate([[0.0], np.diff(out["Minutes"].to_numpy())])
        out["DeltaLicks"] = np.concatenate([[0.0], np.diff(out["CumLicks"].to_numpy())])
        return out[cols].reset_index(drop=True)

    def breaking_point_dfm(self, dfm_id: int) -> dict[int, pd.DataFrame]:
        """:meth:`breaking_point_table` for every chamber of *dfm_id*."""
        return {c: self.breaking_point_table(dfm_id, c) for c in range(1, 7)}

    def plot_breaking_point_dfm(self, dfm_id: int, *, base_font_size: float = 10.0,
                                figsize: tuple[float, float] = (10.5, 6.0)):
        """ΔLicks per light-on period against minutes since training end, one
        panel per chamber, role in the strip."""
        import plotnine as p9

        dfm_id = int(dfm_id)
        frames = []
        for chamber, df in self.breaking_point_dfm(dfm_id).items():
            if df.empty:
                continue
            tmp = df[["Minutes", "DeltaLicks"]].copy()
            r = self.role_of(dfm_id, chamber)
            tmp["Chamber"] = f"Chamber {chamber} ({r.role}, group {r.group})"
            frames.append(tmp)
        if not frames:
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — no breaking-point data")
        data = pd.concat(frames, ignore_index=True)
        data["Chamber"] = pd.Categorical(data["Chamber"],
                                         categories=list(dict.fromkeys(data["Chamber"])),
                                         ordered=True)
        return (p9.ggplot(data, p9.aes("Minutes", "DeltaLicks"))
                + p9.geom_line(size=0.5, color="#2166ac")
                + p9.geom_point(size=1.6, color="#2166ac")
                + p9.facet_wrap("~ Chamber", ncol=3)
                + p9.labs(title=f"DFM {dfm_id} — ΔLicks per light-on period",
                          x="Time since training end (min)", y="ΔLicks")
                + p9.theme_bw(base_size=base_font_size)
                + p9.theme(figure_size=figsize))

    # ------------------------------------------------------------------
    # Summary text and the basic pipeline
    # ------------------------------------------------------------------

    def summary_text(self, *, include_qc: bool = True,
                     qc_data_breaks_multiplier: float = 4.0,
                     qc_bleeding_cutoff: float = 50.0) -> str:
        text = Experiment.summary_text(
            self, include_qc=include_qc,
            qc_data_breaks_multiplier=qc_data_breaks_multiplier,
            qc_bleeding_cutoff=qc_bleeding_cutoff)
        buf = ["", "Progressive Ratio — training by chamber group",
               "--------------------------------------------"]
        table = self.training_table()
        if table.empty:
            buf.append("(no DFMs)")
        else:
            show = table.drop(columns=["Notes"]).copy()
            show["TrainingEndMin"] = show["TrainingEndMin"].map(
                lambda v: "never" if pd.isna(v) else f"{v:.1f}")
            buf.append(show.to_string(index=False))
        warnings = self.training_warnings()
        buf.append("")
        buf.append("Training-flag notes (QC, not errors)")
        buf.append("------------------------------------")
        if warnings:
            buf.extend(f"  {w}" for w in warnings)
        else:
            buf.append("  (all four wells of every group cleared together)")
        buf.append("")
        return text + "\n".join(buf) + "\n"

    def execute_basic_analysis(
        self,
        *,
        data_breaks_multiplier: float = 4.0,
        bleeding_cutoff: float = 50.0,
        range_minutes: Sequence[float] = (0, 0),
        transform_licks: bool | None = None,
        plot_format: str = "png",
        dpi: int = 200,
        skip_qc: bool = False,
    ) -> dict[str, Path]:
        result = TwoWellExperiment.execute_basic_analysis(
            self,
            data_breaks_multiplier=data_breaks_multiplier,
            bleeding_cutoff=bleeding_cutoff,
            range_minutes=range_minutes,
            transform_licks=transform_licks,
            plot_format=plot_format,
            dpi=dpi,
            skip_qc=skip_qc,
        )
        print("\n[PR] Paired-yoked difference table...", flush=True)
        result["paired_yoked_diff"] = self.write_paired_yoked_diff(
            transform_licks=transform_licks)
        print(f"  Done → {result['paired_yoked_diff']}", flush=True)
        print("[PR] Cumulative difference curve data...", flush=True)
        result["pr_cumulative_diff_csv"] = self.write_cumulative_diff()
        print(f"  Done → {result['pr_cumulative_diff_csv']}", flush=True)
        print("[PR] Figures...", flush=True)
        figs = self.write_pr_figures(dpi=dpi)
        result.update(figs)
        print(f"  Done → {len(figs)} figure(s)", flush=True)
        return result


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)

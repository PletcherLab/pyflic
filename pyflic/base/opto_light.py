"""The optogenetic light QC, assembled for one experiment.

:mod:`pyflic.base.opto_qc` judges one DFM from plain arrays; this module feeds
it from an :class:`~pyflic.base.experiment.Experiment` — which DFMs are
optogenetic, what their program says, which chambers each Linkage Group
touches — and turns the verdicts into what the rest of pyflic reads: tables,
per-chamber summary columns, the exclusion list, ``summary.txt`` lines,
``qc/opto/`` files, a QC figure and report blocks.

Every Experiment Type gets it: ``experiment.opto`` is an :class:`OptoLightQC`.
Which DFMs it covers is the ``optogenetics:`` setting — ``yes``, ``no`` or
``auto`` (the default), in ``global:`` or the Project Design, overridable per
DFM in its ``dfms:`` entry.  ``auto`` means a DFM that has a ``Program.txt``
section or any lit LED in its data.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import opto_program as op
from . import opto_qc as oq
from .paradigm_state import GENERATION_DESCRIPTIONS, GENERATION_FLAG, GENERATION_PARADIGM
from .pr_light_qc import VERDICT_FAILED, VERDICT_OK, VERDICT_WARNING

#: ``qc/<QC_SUBDIR>/`` holds the files this module writes.
QC_SUBDIR = "opto"
QC_FILES: dict[str, str] = {
    "opto_light_qc": "opto_light_qc.csv",
    "opto_light_intervals": "opto_light_intervals.csv",
    "opto_light_events": "opto_light_events.csv",
    "opto_program": "opto_program.csv",
    ## Written only when the data have PyMCU's Baseline columns.
    "opto_baselines": "opto_baselines.csv",
}
#: What the light QC adds to every per-chamber summary row of an optogenetic
#: experiment: the chamber's Linkage Groups' flags and their largest
#: unexplained fraction.
ROW_COLUMNS: tuple[str, ...] = ("OptoLightQC", "OptoUnexplainedFraction")

_VERDICT_RANK = {VERDICT_OK: 0, VERDICT_WARNING: 1, VERDICT_FAILED: 2}
_UNEXPLAINED_FLAGS = (oq.UNEXPLAINED, oq.PARTLY_UNEXPLAINED, oq.UNEXPLAINED_NO_PROGRAM)

#: Decision 4's loud warning, for a DFM recorded with firmware 4.0.0
#: (generation 3) that has no ``Program.txt`` section: its rows name each
#: Paradigm but not its type or settings.  ``pyflic lint``, the experiment
#: report's "At a glance" and the Hub's status line give it in these words;
#: ``summary.txt`` says it at more length (:meth:`OptoLightQC.experiment_notes`).
NO_PROGRAM_WARNING = ("firmware 4.0.0 recording without Program.txt: "
                      "Paradigm types and settings unknown")


def no_program_warning(dfm_ids) -> str:
    """:data:`NO_PROGRAM_WARNING` for *dfm_ids*: ``"DFM 1, 2: firmware …"``."""
    return f"DFM {', '.join(str(int(d)) for d in dfm_ids)}: {NO_PROGRAM_WARNING}"


def lacking_program(generations: dict[int, int], program: op.OptoProgram | None) -> list[int]:
    """The DFMs of *generations* (``{dfm: Generation}``) recorded with
    firmware 4.0.0 whose Paradigm types *program* cannot give: all of them
    without a program, else those without a section.  A ``***ALL DFM***``
    export lacks nothing: it says every well was Off."""
    firmware4 = sorted(int(d) for d, g in generations.items() if g == GENERATION_PARADIGM)
    if program is None:
        return firmware4
    if program.grammar == op.GRAMMAR_SIMPLE:
        return []
    return [d for d in firmware4 if program.section(d) is None]


def dfms_without_program(exp) -> list[int]:
    """:func:`lacking_program` for a loaded experiment, over the DFMs its
    light QC covers (as ``summary.txt`` says it); ``[]`` for anything that
    cannot say, so a status line never fails on it."""
    opto = getattr(exp, "opto", None)
    try:
        return list(opto.dfms_without_program()) if opto is not None else []
    except Exception:  # noqa: BLE001 - a status line or a cover must not fail
        return []


def recording_origin(raw: pd.DataFrame | None) -> datetime | None:
    """The clock time of recording minute 0, from the first row's ``Date``,
    ``Time`` and ``MSec`` less its elapsed ``Seconds``; ``None`` for a file
    without dates (a v2 export, a synthetic fixture)."""
    if raw is None or len(raw) == 0 or "Date" not in raw.columns or "Time" not in raw.columns:
        return None
    first = raw.iloc[0]
    time_text = str(first["Time"]).strip()
    text = f"{str(first['Date']).strip()} {time_text}"
    upper = time_text.upper()
    fmt = ("%m/%d/%Y %I:%M:%S %p" if ("AM" in upper or "PM" in upper)
           else "%m/%d/%Y %H:%M:%S")
    try:
        stamp = datetime.strptime(text, fmt)
    except ValueError:
        try:
            stamp = pd.to_datetime(text).to_pydatetime()
        except (ValueError, TypeError):
            return None
    msec = pd.to_numeric(pd.Series([first.get("MSec", 0)]), errors="coerce").iloc[0]
    msec = 0.0 if pd.isna(msec) else float(msec)
    if "Seconds" in raw.columns:
        elapsed = float(pd.to_numeric(pd.Series([first["Seconds"]]), errors="coerce")
                        .fillna(0).iloc[0])
    elif "Minutes" in raw.columns:
        elapsed = 60.0 * float(pd.to_numeric(pd.Series([first["Minutes"]]),
                                             errors="coerce").fillna(0).iloc[0])
    else:
        elapsed = 0.0
    return stamp + timedelta(milliseconds=msec) - timedelta(seconds=elapsed)


def _matrix(frame: pd.DataFrame | None, n: int, dtype) -> np.ndarray:
    """The ``W1..W12`` columns of *frame* as an ``(n, 12)`` array."""
    out = np.zeros((n, op.N_WELLS), dtype=dtype)
    if frame is None:
        return out
    for w in range(1, op.N_WELLS + 1):
        col = f"W{w}"
        if col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce").fillna(0).to_numpy()
            out[:, w - 1] = values[:n].astype(dtype)
    return out


def dfm_inputs(dfm) -> oq.DFMInputs:
    """One loaded :class:`~pyflic.base.dfm.DFM` as the arrays the QC reads."""
    raw = dfm.raw_df
    n = len(raw)
    minutes = pd.to_numeric(raw["Minutes"], errors="coerce").to_numpy(dtype=float)
    activity = _matrix(dfm.lick_df, n, bool) | _matrix(dfm.tasting_df, n, bool)
    signal = np.full((n, op.N_WELLS), np.nan)
    for w in range(1, op.N_WELLS + 1):
        if f"W{w}" in raw.columns:
            signal[:, w - 1] = pd.to_numeric(raw[f"W{w}"], errors="coerce").to_numpy(float)

    def column(name: str):
        if name not in raw.columns:
            return None
        return pd.to_numeric(raw[name], errors="coerce").to_numpy(dtype=float)

    chamber_of_well = {int(w): int(ch.index)
                       for ch in (dfm.chambers or []) for w in ch.wells}
    generation = int(getattr(dfm, "generation", GENERATION_FLAG) or GENERATION_FLAG)
    state = getattr(dfm, "paradigm", None)
    ## A Slice that starts later still has the firmware's baseline, captured
    ## at the start of the run, from the rows its clip kept aside (ADR-0020).
    before = getattr(dfm, "before_range", None)
    return oq.DFMInputs(
        dfm_id=int(dfm.id), minutes=minutes,
        samples_per_second=float(dfm.params.samples_per_second),
        lights=_matrix(dfm.lights_df, n, bool), activity=activity,
        feeding=_matrix(dfm.event_df, n, np.int64), raw=signal,
        chamber_of_well=chamber_of_well, origin=recording_origin(raw),
        data_frequency=column("OptoFreq"), data_pulse_width=column("OptoPW"),
        data_dark=column("Dark"), generation=generation,
        state=state if generation == GENERATION_PARADIGM else None,
        start_minutes=None if before is None else before.start_minutes,
        start_raw=None if before is None else before.start_raw,
        baseline_in_force=getattr(dfm, "baseline_in_force", None))


def _split_flags(text: Any) -> list[str]:
    return [f for f in str(text if text is not None else "").split(", ")
            if f and f.lower() != "nan"]


def _generation_of(row) -> int:
    """A QC row's Generation (1 when the row predates the column)."""
    try:
        value = row["Generation"]
    except (KeyError, IndexError):
        return GENERATION_FLAG
    return int(value) if pd.notna(value) else GENERATION_FLAG


def _group_reasons(result: oq.DFMOptoResult) -> dict[Any, dict[str, str]]:
    """``{group: {flag: words}}`` the QC wrote for flags whose numbers are
    not columns of the QC table (generation 3)."""
    return {g["Group"]: g.get("_reasons") or {} for g in result.groups}


#: The program table's Paradigm settings, in the order a reader is shown them:
#: each type's own order in PyMCU's export.
_SETTING_COLUMNS: tuple[str, ...] = (
    "HoldMs", "OnDelayMs", "MaxOnMs", "ActivationMs", "LockoutMs", "Series",
    "FirstRequirement", "Step", "Scale", "Rate", "Cap", "Ratio", "EventGapMs", "Acclimation",
)


def program_view(program: pd.DataFrame) -> pd.DataFrame:
    """The program table (:func:`opto_qc.program_frame`) for a reader: one
    ``Hold (ms)`` column — the Hold of ``CONTEXT.md``, ``HoldMs`` or else
    ``ActivationMs`` — and the Paradigm's other settings in one ``Settings``
    cell, so that a row shows only what its type has."""
    holds: list[Any] = []
    settings: list[str] = []
    for _, row in program.iterrows():
        hold_column = next((c for c in ("HoldMs", "ActivationMs") if pd.notna(row[c])),
                           None)
        holds.append(row[hold_column] if hold_column else pd.NA)
        parts = []
        for column in _SETTING_COLUMNS:
            value = row[column]
            if column == hold_column or pd.isna(value):
                continue
            text = f"{value:g}" if isinstance(value, float) else str(value)
            parts.append(f"{column} {text}")
        settings.append(", ".join(parts) or "—")
    return pd.DataFrame({
        "DFM": program["DFM"], "Interval": program["Interval"],
        "Minutes": program["DurationMin"], "Paradigm": program["Paradigm"],
        "Type": program["ParadigmType"],
        "Hold (ms)": pd.array(holds, dtype="Int64"), "Settings": settings, "Freq (Hz)": program["FrequencyHz"],
        "Linkage": program["Linkage"], "Trigger wells": program["TriggerWells"],
    })


def verdict_text(flags: Any) -> str:
    """A flag list as one table cell whose first word carries the tone:
    ``ok``, ``warning: …`` or ``failed: …``."""
    names = _split_flags(flags)
    if not names:
        return "ok"
    failing = any(f in oq.FAILING_FLAGS for f in names)
    return ("failed: " if failing else "warning: ") + ", ".join(names)


class OptoLightQC:
    """The optogenetic light QC of one experiment (``experiment.opto``)."""

    def __init__(self, experiment) -> None:
        self.experiment = experiment
        self._results: dict[int, tuple] = {}
        self._lit: dict[int, tuple] = {}

    # ------------------------------------------------------------------
    # Scope
    # ------------------------------------------------------------------

    @property
    def program(self) -> op.OptoProgram | None:
        return getattr(self.experiment, "opto_program", None)

    def settings(self) -> oq.OptoQCSettings:
        return oq.OptoQCSettings.from_constants(self.experiment.global_constants)

    def setting(self) -> str:
        """The experiment's ``optogenetics:`` setting."""
        try:
            return op.normalize_setting(getattr(self.experiment, "optogenetics", None))
        except ValueError:
            return op.SETTING_AUTO

    def _dfm_node(self, dfm_id: int) -> dict | None:
        config = self.experiment.config or {}
        nodes = config.get("dfms", config.get("DFMs")) or []
        if isinstance(nodes, dict):
            nodes = [{"id": int(k), **dict(v)} for k, v in nodes.items()]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            try:
                if int(node.get("id", node.get("ID"))) == int(dfm_id):
                    return node
            except (TypeError, ValueError):
                continue
        return None

    def dfm_setting(self, dfm_id: int) -> str:
        """*dfm_id*'s own ``optogenetics:`` if its ``dfms:`` entry states one,
        else the experiment's."""
        node = self._dfm_node(dfm_id)
        if node is not None and node.get(op.SETTING_KEY) is not None:
            try:
                return op.normalize_setting(node.get(op.SETTING_KEY))
            except ValueError:
                pass
        return self.setting()

    def has_light(self, dfm_id: int) -> bool:
        """Whether any LED was ever lit on *dfm_id*."""
        dfm = self.experiment.dfms[int(dfm_id)]
        entry = self._lit.get(int(dfm_id))
        if entry is not None and entry[0] is dfm:
            return entry[1]
        lights = dfm.lights_df
        lit = False
        if lights is not None and len(lights):
            cols = [f"W{w}" for w in range(1, op.N_WELLS + 1) if f"W{w}" in lights.columns]
            lit = bool(cols) and bool(np.asarray(lights[cols].to_numpy(), dtype=bool).any())
        self._lit[int(dfm_id)] = (dfm, lit)
        return lit

    def dfm_ids(self) -> list[int]:
        """The DFMs the light QC covers: ``yes``, or ``auto`` with a program
        section or a lit LED."""
        out: list[int] = []
        program = self.program
        for dfm_id in sorted(int(d) for d in self.experiment.dfms):
            setting = self.dfm_setting(dfm_id)
            if setting == op.SETTING_NO:
                continue
            if setting == op.SETTING_YES or \
                    (program is not None and program.mentions(dfm_id)) or \
                    self.has_light(dfm_id):
                out.append(dfm_id)
        return out

    @property
    def active(self) -> bool:
        """Whether this is an optogenetic experiment at all."""
        return bool(self.dfm_ids())

    # ------------------------------------------------------------------
    # One DFM
    # ------------------------------------------------------------------

    def result(self, dfm_id: int) -> oq.DFMOptoResult:
        """The QC of *dfm_id*, cached until its DFM object, the settings or its
        ``optogenetics:`` setting change (a QC Viewer recompute swaps in DFMs
        re-detected under other parameters, and their licks differ)."""
        dfm_id = int(dfm_id)
        dfm = self.experiment.dfms[dfm_id]
        settings = self.settings()
        setting = self.dfm_setting(dfm_id)
        entry = self._results.get(dfm_id)
        if entry is not None and entry[0] is dfm and entry[1] == settings \
                and entry[2] == setting:
            return entry[3]
        result = oq.analyze_dfm(dfm_inputs(dfm), program=self.program, setting=setting,
                                settings=settings)
        self._results[dfm_id] = (dfm, settings, setting, result)
        return result

    def decay_samples(self, dfm_id: int) -> np.ndarray:
        """Per sample of *dfm_id*, the light decay in samples: the program's
        interval decay where one applies, else ``opto_default_decay_ms``.
        Available whether or not the QC covers the DFM — the Progressive Ratio
        light QC reads it to decide which light events are lick-free."""
        dfm_id = int(dfm_id)
        if dfm_id in self.dfm_ids():
            return self.result(dfm_id).decay_samples
        dfm = self.experiment.dfms[dfm_id]
        minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").to_numpy(float)
        state = (dfm.paradigm if getattr(dfm, "generation", GENERATION_FLAG)
                 == GENERATION_PARADIGM else None)
        return oq.decay_profile(dfm_id, minutes, float(dfm.params.samples_per_second),
                                recording_origin(dfm.raw_df), self.program,
                                self.settings(), state=state)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def _treatments(self, dfm_id: int, chambers) -> str:
        snapshot = self.experiment._opto_design_snapshot()
        names = [snapshot.get((int(dfm_id), int(c))) for c in chambers]
        return ", ".join(dict.fromkeys(n for n in names if n))

    def qc_table(self) -> pd.DataFrame:
        """One row per Linkage Group of every DFM the QC covers —
        ``qc/opto/opto_light_qc.csv``.

        ``Flags`` lists what fired: *unexplained light*, *light while off*,
        *no light recorded* and, on a firmware 4.0.0 recording, *broken
        yoking*, *paradigm not in program* and the rule checks that fail
        (*wrong activation length*, *lockout broken*, *requirement series
        wrong*, *max on exceeded*, *short activations*) fail a group
        (:data:`oq.FAILING_FLAGS`); the rest are warnings.  ``Verdict``
        is ``ok``, ``warning`` or ``failed``; ``Excluded`` says a failed
        group's chambers leave the analysis (``exclude_failed_opto_chambers``,
        off by default).  The rule checks' counts are nullable integers.
        The definitions are in :mod:`pyflic.base.opto_qc` and
        :mod:`pyflic.base.rule_checks`.
        """
        settings = self.settings()
        rows = []
        for dfm_id in self.dfm_ids():
            for group in self.result(dfm_id).groups:
                row = {k: v for k, v in group.items() if not k.startswith("_")}
                row["Treatment"] = self._treatments(dfm_id, group["_chambers"])
                row["Excluded"] = bool(settings.exclude
                                       and row["Verdict"] == VERDICT_FAILED
                                       and row["Treatment"])
                rows.append(row)
        frame = pd.DataFrame(rows, columns=list(oq.OPTO_QC_COLUMNS))
        for column in oq.OPTO_QC_INTEGER_COLUMNS:
            frame[column] = pd.array(pd.to_numeric(frame[column]), dtype="Int64")
        return frame

    def interval_table(self) -> pd.DataFrame:
        rows = [r for d in self.dfm_ids() for r in self.result(d).intervals]
        return pd.DataFrame(rows, columns=list(oq.OPTO_INTERVAL_COLUMNS))

    def events_table(self) -> pd.DataFrame:
        """One row per light event of every DFM the QC covers —
        ``qc/opto/opto_light_events.csv`` (:data:`oq.OPTO_EVENT_COLUMNS`).
        The ledger's millisecond columns are nullable integers and
        ``Overlap`` a nullable boolean, ``NA`` on the ``OptoCol1`` rows."""
        rows = [r for d in self.dfm_ids() for r in self.result(d).events]
        frame = pd.DataFrame(rows, columns=list(oq.OPTO_EVENT_COLUMNS))
        for column in oq.OPTO_EVENT_INTEGER_COLUMNS:
            frame[column] = pd.array(pd.to_numeric(frame[column]), dtype="Int64")
        frame["Overlap"] = pd.array(frame["Overlap"].astype(object).where(
            frame["Overlap"].notna(), None), dtype="boolean")
        return frame

    def program_table(self) -> pd.DataFrame:
        """The program as read, one row per interval of every DFM the QC
        covers — ``qc/opto/opto_program.csv`` (:data:`oq.OPTO_PROGRAM_COLUMNS`)."""
        return oq.program_frame(self.program, self.dfm_ids())

    def baselines_table(self) -> pd.DataFrame:
        """How each well's Baseline in force moved, one row per well of every
        DFM whose data have PyMCU's Baseline columns —
        ``qc/opto/opto_baselines.csv`` (:data:`oq.OPTO_BASELINE_COLUMNS`).
        Empty when no DFM's data have them."""
        rows = [r for d in self.dfm_ids() for r in self.result(d).baselines]
        return pd.DataFrame(rows, columns=list(oq.OPTO_BASELINE_COLUMNS))

    # ------------------------------------------------------------------
    # Per chamber
    # ------------------------------------------------------------------

    def chamber_verdicts(self) -> dict[tuple[int, int], dict]:
        """``{(dfm, chamber): {"verdict", "flags", "fraction"}}`` — each
        chamber gets the worst verdict, the union of the flags and the largest
        unexplained fraction of the Linkage Groups its wells belong to."""
        out: dict[tuple[int, int], dict] = {}
        for dfm_id in self.dfm_ids():
            for group in self.result(dfm_id).groups:
                flags = _split_flags(group["Flags"])
                fraction = group["UnexplainedFraction"]
                for chamber in group["_chambers"]:
                    key = (int(dfm_id), int(chamber))
                    cur = out.get(key)
                    if cur is None:
                        out[key] = {"verdict": group["Verdict"], "flags": list(flags),
                                    "fraction": fraction}
                        continue
                    if _VERDICT_RANK[group["Verdict"]] > _VERDICT_RANK[cur["verdict"]]:
                        cur["verdict"] = group["Verdict"]
                    cur["flags"] = list(dict.fromkeys([*cur["flags"], *flags]))
                    values = [v for v in (cur["fraction"], fraction) if pd.notna(v)]
                    cur["fraction"] = max(values) if values else np.nan
        return out

    def failed_chambers(self) -> dict[tuple[int, int], list[str]]:
        """``{(dfm, chamber): failing flags}`` for the chambers auto-removal
        takes out — empty unless ``exclude_failed_opto_chambers`` is on."""
        if not self.settings().exclude:
            return {}
        return {key: [f for f in v["flags"] if f in oq.FAILING_FLAGS]
                for key, v in self.chamber_verdicts().items()
                if v["verdict"] == VERDICT_FAILED}

    def with_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """*df* (per-chamber rows with ``DFM`` and ``Chamber``) with
        :data:`ROW_COLUMNS`, redone on every call: a summary read back from the
        disk cache carries whatever they said when it was written."""
        if df is None or df.empty or "DFM" not in df.columns or "Chamber" not in df.columns:
            return df
        stale = [c for c in ROW_COLUMNS if c in df.columns]
        if stale:
            df = df.drop(columns=stale)
        if not self.active:
            return df
        verdicts = self.chamber_verdicts()
        keys = list(zip(pd.to_numeric(df["DFM"]).astype(int),
                        pd.to_numeric(df["Chamber"]).astype(int)))
        df = df.copy()
        df["OptoLightQC"] = [", ".join(verdicts[k]["flags"]) if k in verdicts else ""
                             for k in keys]
        df["OptoUnexplainedFraction"] = [verdicts[k]["fraction"] if k in verdicts
                                         else np.nan for k in keys]
        return df

    # ------------------------------------------------------------------
    # Words
    # ------------------------------------------------------------------

    def program_line(self) -> str:
        """Where the program came from, in one line."""
        exp = self.experiment
        program = self.program
        error = getattr(exp, "opto_program_error", None)
        if error:
            return f"Program.txt could not be used: {error}"
        if program is None:
            return ("no Program.txt in data/ — the light QC is limited (linkage inferred "
                    "from wells lit together, decay assumed; see each DFM)")
        name = program.path.name if program.path is not None else "Program.txt"
        if program.grammar == op.GRAMMAR_SIMPLE:
            return (f"{name}: the recording was made without a Program (***ALL DFM***), "
                    f"every well Off, so any light is a fault")
        start = program.start.strftime("%Y-%m-%d %H:%M:%S") if program.start else "no start"
        duration = (f"{program.duration_min:g} min" if program.duration_min is not None
                    else "no duration")
        return (f"{name}: {len(program.dfms)} DFM section(s), start {start}, {duration}, "
                f"firmware baseline {'on' if program.baseline else 'off'}")

    def generations(self) -> dict[int, int]:
        """``{dfm: Generation}`` for every DFM the QC covers."""
        out: dict[int, int] = {}
        for dfm_id in self.dfm_ids():
            dfm = self.experiment.dfms[dfm_id]
            out[dfm_id] = int(getattr(dfm, "generation", GENERATION_FLAG) or GENERATION_FLAG)
        return out

    def dfms_without_program(self) -> list[int]:
        """The covered DFMs recorded with firmware 4.0.0 that have no
        ``Program.txt`` section (:func:`lacking_program`), for
        :data:`NO_PROGRAM_WARNING`."""
        return lacking_program(self.generations(), self.program)

    def experiment_notes(self) -> list[str]:
        """Notes about the experiment as a whole: the program's parse warnings,
        what the loader derived from it, a program whose paradigm is not the
        Experiment Type's, the DFMs' Generations when any is not the first,
        and a firmware 4.0.0 recording without its program."""
        exp = self.experiment
        notes: list[str] = []
        program = self.program
        if program is not None:
            skipped = program.skipped
            shown = skipped[:8]
            notes.extend(f"Program.txt: {w}" for w in shown)
            if len(skipped) > len(shown):
                notes.append(f"Program.txt: and {len(skipped) - len(shown)} "
                             f"more line(s) skipped")
            if program.grammar == op.GRAMMAR_SIMPLE:
                notes.append(f"Program.txt: {op.SIMPLE_WARNING}")
            sections = {d: program.dfms[d] for d in self.dfm_ids() if d in program.dfms}
            if sections:
                paradigms = oq.paradigms_in(op.OptoProgram(
                    path=None, start=None, end=None, duration_min=None, baseline=True,
                    dfms=sections))
                etype = getattr(exp, "experiment_type", None)
                has = getattr(etype, "has_module", None)
                is_pr = bool(has("ratio")) if callable(has) else False
                display = getattr(etype, "display_name", "Custom")
                ## The Progressive Ratio module covers both ratio Paradigms.
                ratios = [p for p in op.RATIO_PARADIGMS if p in paradigms]
                if ratios and not is_pr:
                    names = " and a ".join(ratios)
                    notes.append(f"Program.txt runs a {names}, but the experiment "
                                 f"type {display} has no Progressive Ratio module: the "
                                 f"breaking point and the paired / yoked analysis need a "
                                 f"type that includes it, such as ProgressiveRatio")
                if is_pr and not ratios:
                    listed = ", ".join(sorted(paradigms)) or "no triggered interval"
                    notes.append(f"the experiment type {display} runs the Progressive "
                                 f"Ratio module, but Program.txt runs no progressive-ratio "
                                 f"or fixed-ratio interval ({listed})")
        generations = self.generations()
        if any(g != GENERATION_FLAG for g in generations.values()):
            by_generation: dict[int, list[int]] = {}
            for dfm_id, g in generations.items():
                by_generation.setdefault(g, []).append(dfm_id)
            notes.append("DFM generations: " + "; ".join(
                f"DFM {', '.join(str(d) for d in ids)} generation {g} "
                f"({GENERATION_DESCRIPTIONS.get(g, 'unknown')})"
                for g, ids in sorted(by_generation.items())))
        lacking = lacking_program(generations, program)
        if lacking and program is None:
            notes.append(f"no Program.txt for DFM {', '.join(str(d) for d in lacking)}, "
                         f"recorded with DFM firmware 4.0.0: the rows name each Paradigm "
                         f"but not its type, settings or thresholds, so the type-dependent "
                         f"light checks are not made; copy the Program.txt PyMCU exported "
                         f"for this recording into data/")
        elif lacking:
            notes.append(f"Program.txt has no usable section for DFM "
                         f"{', '.join(str(d) for d in lacking)}, recorded with DFM "
                         f"firmware 4.0.0: the type-dependent light checks are not "
                         f"made there")
        notes.extend(getattr(exp, "program_notes", None) or [])
        return notes

    def lines(self) -> list[str]:
        """One plain-language block per DFM with something to say, then one
        per flagged Linkage Group — for the summary, the Hub's log and the
        report."""
        table = self.qc_table()
        lines: list[str] = []
        for dfm_id in self.dfm_ids():
            result = self.result(dfm_id)
            dfm_flags = [f for f in result.flags]
            extra = _group_reasons(result)
            if dfm_flags or result.notes:
                head = f"DFM {dfm_id}"
                if dfm_flags:
                    head += f": {verdict_text(', '.join(dfm_flags))}"
                lines.append(head)
                lines.extend(f"    · {note}" for note in result.notes)
            rows = table[table["DFM"] == dfm_id]
            for _, r in rows.iterrows():
                own = [f for f in _split_flags(r["Flags"]) if f not in dfm_flags]
                reasons = [self._reason(flag, r, extra.get(r["Group"])) for flag in own]
                where = (f"DFM {dfm_id} linkage group {r['Group']} ({r['Wells']}; "
                         f"chamber(s) {r['Chambers']}"
                         + (f"; {r['Treatment']}" if r["Treatment"] else "") + ")")
                if r["Verdict"] == VERDICT_FAILED and any(f in oq.FAILING_FLAGS for f in own):
                    outcome = ("EXCLUDED (exclude_failed_opto_chambers)" if r["Excluded"]
                               else "FAILED, kept and flagged "
                                    "(exclude_failed_opto_chambers is off)")
                elif reasons:
                    outcome = "warning, kept"
                else:
                    outcome = ""
                if reasons:
                    lines.append(f"{where}: {outcome}")
                    lines.extend(f"    - {reason}" for reason in reasons)
                if r["Notes"]:
                    if not reasons:
                        lines.append(f"{where}:")
                    lines.extend(f"    · {note}" for note in str(r["Notes"]).split("; "))
        return lines

    def group_lines(self, dfm_id: int, group: int) -> list[str]:
        """What the light QC says about one Linkage Group: its verdict, a line
        per flag, its notes, and the notes and flags of its DFM — for the QC
        Viewer's detail pane."""
        table = self.qc_table()
        rows = table[(table["DFM"] == int(dfm_id)) & (table["Group"] == int(group))]
        if rows.empty:
            return []
        r = rows.iloc[0]
        result = self.result(dfm_id)
        verdict = verdict_text(r["Flags"])
        if r["Excluded"]:
            verdict += " (excluded by exclude_failed_opto_chambers)"
        out = [f"Verdict: {verdict}"]
        own = [f for f in _split_flags(r["Flags"]) if f not in result.flags]
        extra = _group_reasons(result).get(r["Group"])
        out += [f"- {self._reason(flag, r, extra)}" for flag in own]
        if r["Notes"]:
            out += [f"· {note}" for note in str(r["Notes"]).split("; ")]
        if result.flags or result.notes:
            head = f"DFM {int(dfm_id)}"
            if result.flags:
                head += f": {verdict_text(', '.join(result.flags))}"
            out += ["", head, *[f"· {note}" for note in result.notes]]
        return out

    @staticmethod
    def _reason(flag: str, r, extra: dict[str, str] | None = None) -> str:
        """One flag of a QC row in words, with its numbers: from the row's
        columns, or for the generation-3 checks whose numbers have no column,
        the words the QC wrote for the group (*extra*)."""
        if extra and flag in extra:
            return extra[flag]
        if flag == oq.ACCLIMATION_COUNT_MISMATCH:
            return (f"{flag}: {r['AcclimationActivations']:.0f} Activations in completed "
                    f"Acclimation, the Paradigm's Acclimation adds up to "
                    f"{r['AcclimationExpected']:.0f}")
        if flag == oq.EVENT_DISAGREEMENT:
            return (f"{flag}: {r['FirmwareEvents']:.0f} Firmware Events and "
                    f"{r['PyflicEvents']:.0f} pyflic feeding events under the ratio "
                    f"Paradigm's Main phase")
        firmware4 = _generation_of(r) == GENERATION_PARADIGM
        if flag in _UNEXPLAINED_FLAGS:
            window = ("the Hold, or at the onset of a fixed-length Activation" if firmware4
                      else "the decay")
            text = (f"{flag}: {r['UnexplainedSec']:.0f} s of {r['JudgedLitSec']:.0f} s lit "
                    f"({100 * r['UnexplainedFraction']:.0f}%) with no trigger-well lick or "
                    f"touch within {window}")
            if pd.notna(r["UnexplainedOnsetMin"]):
                text += f", from minute {r['UnexplainedOnsetMin']:.0f}"
            if r["LikelyCause"]:
                text += f"; {r['LikelyCause']}"
            if firmware4 and r["LikelyCause"] in oq.CONTACT_CAUSES:
                text += (f" ({r['ContactWithoutActivitySec']:.0f} s lit by Activations and "
                         f"{r['FirmwareEventsWithoutActivity']:.0f} Firmware Event(s) "
                         f"with no lick or touch)")
            return text
        if flag == oq.LIGHT_WHILE_OFF:
            return (f"light while off: lit {r['LitWhileOffSec']:.1f} s while every "
                    f"member's threshold was {'Off' if firmware4 else '-1'}")
        if flag == oq.NO_LIGHT_EVENTS:
            return (f"no light events: {int(r['FeedingEvents'])} feeding bout(s) at its "
                    f"trigger wells and never lit")
        if flag == oq.UNLIT_FEEDING:
            return (f"unlit feeding: {int(r['UnlitFeedingEvents'])} of "
                    f"{int(r['ClosedLoopFeedingEvents'])} closed-loop feeding bouts at its "
                    f"trigger wells never lit it")
        if flag == oq.OPEN_LOOP_DARK:
            return (f"open loop not lit: lit for {100 * r['OpenLoopLitFraction']:.0f}% of "
                    f"its open-loop time")
        return flag

    def summary_lines(self) -> list[str]:
        """The ``summary.txt`` section."""
        settings = self.settings()
        out = ["", "Optogenetic light QC", "--------------------",
               "Was the light where the licks were?  The firmware lights a linkage group",
               "from its own reading of the group's trigger wells during the run; pyflic",
               "counts licks afterwards.  A lit sample is explained when a trigger well has",
               "a lick or a touch within the interval's light decay, plus "
               f"{settings.tolerance_samples} sample(s), before it.",
               "Computed over the whole recording, whatever window a table uses.",
               f"Program: {self.program_line()}",
               f"Settings: {settings.describe()}"]
        firmware4 = GENERATION_PARADIGM in self.generations().values()
        if firmware4:
            out += [f"Firmware 4.0.0 checks: {settings.describe_firmware4()}",
                    "On a firmware 4.0.0 recording the light is OnMs and each trigger-well",
                    "Activation owns it: a fixed-length Activation (FixedInterval, ratio Main)",
                    "is explained when its well had a lick or touch within the tolerance of",
                    "its onset; closed-loop and Acclimation light keep the decay rule with",
                    "the Hold.  Each Paradigm is also checked against its own rule, one",
                    "Activation at a time, with Program.txt's settings (the rule checks)."]
        out.append("")
        program = self.program_table()
        if not program.empty:
            show = program_view(program)[["DFM", "Interval", "Minutes", "Paradigm", "Type",
                                          "Hold (ms)", "Settings", "Freq (Hz)",
                                          "Trigger wells"]]
            show.columns = ["DFM", "Int", "Min", "Paradigm", "Type", "Hold", "Settings",
                            "Hz", "Trigger wells"]
            out.append(show.to_string(index=False))
            out.append("")
        table = self.qc_table()
        if table.empty:
            return out + ["(no DFM is covered)", ""]
        roles = firmware4 and table["Roles"].notna().any()
        columns = ["DFM", "Group", "Wells", "Chambers", *(["Roles"] if roles else []),
                   "Treatment", "LitSec", "UnexplainedFraction", "UnexplainedOnsetMin",
                   "ContactWithoutActivitySec", "Verdict"]
        show = table[columns].copy()
        show.columns = ["DFM", "Group", "Wells", "Chambers", *(["Roles"] if roles else []),
                        "Treatment", "Lit(s)", "Unexpl", "Onset", "Contact(min)", "Verdict"]
        if roles:
            show["Roles"] = show["Roles"].map(lambda v: "—" if pd.isna(v) else v)
        show["Lit(s)"] = show["Lit(s)"].map(lambda v: f"{v:.0f}")
        show["Unexpl"] = show["Unexpl"].map(lambda v: "—" if pd.isna(v) else f"{100 * v:.0f}%")
        show["Onset"] = show["Onset"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")
        show["Contact(min)"] = show["Contact(min)"].map(
            lambda v: "—" if pd.isna(v) else f"{v / 60.0:.0f}")
        out.append(show.to_string(index=False))
        out.append("(Contact: minutes the emulated firmware trigger saw the signal above "
                   "threshold with no pyflic lick or touch.)")
        if firmware4:
            out.append("(On a firmware 4.0.0 DFM, Contact is the minutes lit by "
                       "Activations whose onset had no pyflic lick or touch.)")
        notes = self.experiment_notes()
        detail = self.lines()
        if notes or detail:
            out.append("")
        out.extend(f"  {note}" for note in notes)
        out.extend(f"  {line}" for line in detail)
        excluded = table[table["Excluded"]]
        failed_kept = table[(table["Verdict"] == VERDICT_FAILED) & ~table["Excluded"]]
        out.append("")
        if not excluded.empty:
            groups = ", ".join(f"DFM {int(r.DFM)} group {r.Group}" for r in excluded.itertuples())
            out.append(f"Excluded by the opto light QC (every chamber of each): {groups}.")
        if not failed_kept.empty:
            groups = ", ".join(f"DFM {int(r.DFM)} group {r.Group}"
                               for r in failed_kept.itertuples())
            out.append(f"FAILED but kept and flagged (exclude_failed_opto_chambers is off): "
                       f"{groups}.")
        if excluded.empty and failed_kept.empty:
            out.append("No linkage group failed the opto light QC.")
        out.append("")
        return out

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    def write(self, out_dir: str | Path | None = None) -> dict[str, Path]:
        """Write ``qc/opto/``: the per-group verdicts, the per-interval and
        per-event tables and the program as read."""
        if out_dir is None:
            qc_dir = self.experiment.qc_dir
            if qc_dir is None:
                raise ValueError("experiment_dir must be set to write the opto light QC.")
            out_dir = Path(qc_dir) / QC_SUBDIR
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        tables = {
            "opto_light_qc": self.qc_table(),
            "opto_light_intervals": self.interval_table(),
            "opto_light_events": self.events_table(),
            "opto_program": self.program_table(),
        }
        ## The baselines table only when PyMCU wrote Baseline columns.
        baselines = self.baselines_table()
        if not baselines.empty:
            tables["opto_baselines"] = baselines
        written: dict[str, Path] = {}
        for key, frame in tables.items():
            path = out / QC_FILES[key]
            frame.to_csv(path, index=False, na_rep="NA")
            written[key] = path
        return written

    # ------------------------------------------------------------------
    # The QC figure
    # ------------------------------------------------------------------

    def plot_dfm(self, dfm_id: int, *, binsize_min: float = 10.0,
                 base_font_size: float = 10.0, figsize: tuple[float, float] | None = None,
                 groups=None):
        """Light explained over time (QC) for one DFM, one row per Linkage
        Group that was ever lit or touched.

        The left panel of a row is the group's lit time per bin, stacked:
        explained (a trigger-well lick or touch within the decay), unexplained,
        lit while every threshold was -1, and not judged by licks (open loop,
        non-feeding activation).  The right one, when the program allows the
        Emulated Trigger, is the time the firmware would have read the trigger
        wells above threshold while pyflic saw no lick or touch — the
        drifting-baseline signature.  On a firmware 4.0.0 recording the lit
        time is ``OnMs`` (milliseconds, not samples) and the right panel is
        the light of Activations the firmware counted with no lick or touch at
        their onset.  A group never lit or touched is left out unless it was
        flagged.  *groups* limits the figure to those Linkage Groups, drawn
        whatever they hold (the QC Viewer shows one at a time).
        """
        import plotnine as p9

        dfm_id = int(dfm_id)
        if dfm_id not in self.dfm_ids():
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — not covered by the "
                                               f"opto light QC")
        result = self.result(dfm_id)
        binsize = max(float(binsize_min), 1.0)
        emulated = any(pd.notna(g["ContactSec"]) for g in result.groups)
        drift_kind = ("firmware events, no lick"
                      if result.generation == GENERATION_PARADIGM
                      else "firmware contact, no lick")
        frames, order = [], []
        wanted = None if groups is None else {int(g) for g in groups}
        for group in result.groups:
            tl = result.timelines.get(group["Group"])
            if tl is None or tl.empty:
                continue
            if wanted is not None:
                if int(group["Group"]) not in wanted:
                    continue
            elif not self._drawn(group, tl):
                continue
            head = (f"Group {group['Group']} ({group['Wells']}) — "
                    f"{verdict_text(group['Flags'])}")
            light_panel = f"{head}\nlight (s per bin)"
            contact_panel = f"{head}\n{drift_kind} (s per bin)"
            order.append(light_panel)
            if emulated:
                order.append(contact_panel)
            bins = np.floor(tl["Minute"].to_numpy() / binsize) * binsize
            agg = tl.assign(Bin=bins).groupby("Bin", as_index=False)[
                ["LitSec", "ExplainedSec", "UnexplainedSec", "LitWhileOffSec",
                 "NotJudgedSec", "ContactWithoutActivitySec"]].sum()
            minute = agg["Bin"] + binsize / 2.0
            for column, kind in (("ExplainedSec", "explained"),
                                 ("UnexplainedSec", "unexplained"),
                                 ("LitWhileOffSec", "lit while off"),
                                 ("NotJudgedSec", "not judged by licks")):
                frames.append(pd.DataFrame({"Minute": minute, "Panel": light_panel,
                                            "Seconds": agg[column], "Kind": kind}))
            if emulated:
                frames.append(pd.DataFrame({"Minute": minute, "Panel": contact_panel,
                                            "Seconds": agg["ContactWithoutActivitySec"],
                                            "Kind": drift_kind}))
        if not frames:
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — no light to show")
        subject = f"DFM {dfm_id}"
        if wanted is not None and len(wanted) == 1:
            subject += f" group {next(iter(wanted))}"
        data = pd.concat(frames, ignore_index=True)
        data["Panel"] = pd.Categorical(data["Panel"], categories=order, ordered=True)
        kinds = ["explained", "unexplained", "lit while off", "not judged by licks",
                 drift_kind]
        data["Kind"] = pd.Categorical(data["Kind"], categories=kinds, ordered=True)
        ncol = 2 if emulated else min(3, len(order))
        nrow = -(-len(order) // ncol)
        if figsize is None:
            figsize = (4.6 * ncol, min(24.0, 1.9 * nrow + 1.0))
        colors = {"explained": "#9DB4C8", "unexplained": "#D62728",
                  "lit while off": "#6A3D9A", "not judged by licks": "#D9D9D9",
                  drift_kind: "#E69F00"}
        g = (p9.ggplot(data, p9.aes("Minute", "Seconds", fill="Kind"))
             + p9.geom_col(width=binsize, position="stack")
             ## Every panel from zero, and an empty one a unit tall rather
             ## than centred on nothing.
             + p9.expand_limits(y=[0, 1])
             + p9.facet_wrap("~ Panel", ncol=ncol, scales="free_y")
             + p9.scale_fill_manual(values=colors, drop=False)
             + p9.labs(title=f"{subject} — light explained by trigger-well licks "
                             f"({binsize:g}-min bins)",
                       x="Minutes", y="Seconds per bin", fill="")
             + p9.theme_bw(base_size=base_font_size)
             + p9.theme(figure_size=figsize, legend_position="bottom"))
        return g

    def plot_rows(self, dfm_id: int) -> int:
        """How many rows :meth:`plot_dfm` draws for *dfm_id*, for sizing its box."""
        if int(dfm_id) not in self.dfm_ids():
            return 1
        result = self.result(dfm_id)
        emulated = any(pd.notna(g["ContactSec"]) for g in result.groups)
        panels = 0
        for group in result.groups:
            tl = result.timelines.get(group["Group"])
            if tl is not None and not tl.empty and self._drawn(group, tl):
                panels += 2 if emulated else 1
        ncol = 2 if emulated else 3
        return max(1, -(-panels // ncol))

    @staticmethod
    def _drawn(group: dict, timeline: pd.DataFrame) -> bool:
        """Whether :meth:`plot_dfm` draws *group*: it was lit or touched, or it
        was flagged — a group whose LED never lit is exactly what to show."""
        return bool(timeline["LitSec"].sum() > 0
                    or timeline["ContactWithoutActivitySec"].sum() > 0
                    or _split_flags(group["Flags"]))

    def write_figures(self, out_dir: str | Path | None = None, *, dpi: int = 200
                      ) -> dict[str, Path]:
        """``qc/opto/opto_light_dfm<id>.png`` for every covered DFM."""
        if out_dir is None:
            qc_dir = self.experiment.qc_dir
            if qc_dir is None:
                raise ValueError("experiment_dir must be set to write figures.")
            out_dir = Path(qc_dir) / QC_SUBDIR
        out = Path(out_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for dfm_id in self.dfm_ids():
            path = out / f"opto_light_dfm{dfm_id}.png"
            self.plot_dfm(dfm_id).save(str(path), dpi=dpi, verbose=False)
            written[f"opto_light_dfm{dfm_id}"] = path
        return written

    # ------------------------------------------------------------------
    # The experiment report
    # ------------------------------------------------------------------

    def report_glance_blocks(self) -> list[Any]:
        """The cover's callout: how many Linkage Groups failed or warned, and
        how many groups each failing and each warning flag was raised on
        (the rule checks' among them, on a firmware 4.0.0 recording)."""
        from . import report_layout as rl

        if not self.active:
            return []
        table = self.qc_table()
        n = len(table)
        failed = table[table["Verdict"] == VERDICT_FAILED]
        warned = table[table["Verdict"] == VERDICT_WARNING]
        name = lambda rows: "; ".join(f"DFM {int(r.DFM)} group {r.Group}"  # noqa: E731
                                      for r in rows.itertuples())
        firmware4 = GENERATION_PARADIGM in self.generations().values()
        if not failed.empty:
            excluded = int(failed["Excluded"].sum())
            text = (f"{len(failed)} of {n} linkage groups failed ({name(failed)}): "
                    f"{_flag_counts(failed, oq.FAILING_FLAGS)}"
                    + (f"; {excluded} excluded." if excluded else
                       "; kept and flagged, because exclude_failed_opto_chambers is off."))
            tone = "failed"
        elif firmware4:
            text = (f"Every linkage group's light was explained by its licks and kept its "
                    f"Paradigm's rule ({n} of {n}).")
            tone = "ok"
        else:
            text = f"Every linkage group's light was explained by its licks ({n} of {n})."
            tone = "ok"
        if not warned.empty:
            text += (f"  {len(warned)} warning(s), {name(warned)}: "
                     f"{_flag_counts(warned, oq.WARNING_FLAGS)} — see Quality control.")
            tone = "warning" if tone == "ok" else tone
        if self.program is None:
            text += "  No Program.txt: the check is limited."
            tone = "warning" if tone == "ok" else tone
        return [rl.Callout(text, tone=tone,
                           title="Opto light QC — was the light where the licks were?")]

    def report_qc_blocks(self) -> list[Any]:
        """The Quality control subsection, for every Experiment Type."""
        from . import report_layout as rl

        if not self.active:
            return []
        from . import rule_figures

        settings = self.settings()
        generations = self.generations()
        firmware4 = GENERATION_PARADIGM in generations.values()
        rules_error = None
        if firmware4:
            ## Generation 3 records each Activation's light, and each Paradigm's
            ## rule is checked below; generations 1 and 2 keep the decay rule.
            try:
                rules = rule_figures.rule_sections(self)
            except Exception as exc:  # noqa: BLE001 - the subsections, not the report
                rules, rules_error = [], exc
            intro = (
                "The firmware lights a linkage group from its own reading of the group's "
                "trigger wells during the run; pyflic counts licks afterwards, from the "
                "baselined signal.  On a firmware 4.0.0 recording each Activation's light is "
                "recorded to the millisecond: a fixed-length Activation is explained when "
                "its trigger well had a lick or a touch at its onset, closed-loop light when "
                "one came within the Hold, plus "
                f"{settings.tolerance_samples} sample(s), before it.  Unexplained light — "
                "light with no lick behind it — is the sign of a drifting sensor, a "
                "sustained contact or a hardware fault.  "
                + ("Each Paradigm is also checked against its own rule, one Activation at a "
                   "time, with Program.txt's settings: one subsection per Paradigm follows "
                   "the figures.  " if rules or rules_error is not None else
                   "Each Paradigm's own rule is checked only with Program.txt's settings, "
                   "and this recording has none for its firmware 4.0.0 DFMs.  ")
                + "Computed over the whole recording.")
        else:
            rules = []
            intro = (
                "The firmware lights a linkage group from its own reading of the group's "
                "trigger wells during the run; pyflic counts licks afterwards, from the "
                "baselined signal.  A lit sample is explained when a trigger well has a "
                "lick or a touch within the interval's light decay, plus "
                f"{settings.tolerance_samples} sample(s), before it.  Unexplained light — "
                "light with no lick behind it — is the sign of a drifting sensor, a "
                "sustained contact or a hardware fault.  Computed over the whole recording.")
        blocks: list[Any] = [rl.Heading("Optogenetic light QC", level=2), rl.Paragraph(intro)]
        older = sorted(d for d, g in generations.items() if g != GENERATION_PARADIGM)
        if older:
            ## Decision 3 of docs/paradigm-analyses-plan.md: name the checks
            ## that did not run, and why.
            which = (f"DFM {older[0]}" if len(older) == 1
                     else "DFMs " + ", ".join(str(d) for d in older[:-1]) + f" and {older[-1]}")
            blocks.append(rl.Paragraph(
                f"The firmware 4.0.0 rule checks (short activations, max on exceeded, wrong "
                f"activation length, lockout broken, contact without activation, requirement "
                f"series wrong, non-feeding dark, interval started late and DFM restarted) did "
                f"not run for {which}: {'its' if len(older) == 1 else 'their'} data come from "
                f"older DFM firmware, which records only which LEDs were lit at each 200 ms "
                f"record, and that cannot time a light period.  The checks above did run.",
                size=rl.SIZE_SMALL, color=rl.MUTED))
        program_tone = "info" if self.program is not None else "warning"
        blocks.append(rl.Callout(self.program_line(), tone=program_tone, title="Program"))
        program = self.program_table()
        if not program.empty:
            blocks.append(rl.Table(program_view(program),
                                   caption="The program, as the MCU echoed it",
                                   formats={"Minutes": "{:g}"}))
        table = self.qc_table()
        columns: dict[str, Any] = {
            "DFM": table["DFM"], "Group": table["Group"], "Wells": table["Wells"],
            "Chambers": table["Chambers"]}
        if firmware4 and table["Roles"].notna().any():
            columns["Roles"] = table["Roles"].map(lambda v: "—" if pd.isna(v) else v)
        columns.update({
            "Treatment": table["Treatment"],
            "Lit (s)": table["LitSec"],
            "Unexplained": table["UnexplainedFraction"],
            "From (min)": table["UnexplainedOnsetMin"],
            "Contact, no lick (min)": table["ContactWithoutActivitySec"] / 60.0,
            "Verdict": [("excluded" if ex else verdict_text(f))
                        for f, ex in zip(table["Flags"], table["Excluded"])],
        })
        view = pd.DataFrame(columns)
        thresholds = f"Thresholds: {settings.describe()}."
        if firmware4:
            thresholds += f"  Firmware 4.0.0 checks: {settings.describe_firmware4()}."
        blocks += [
            rl.Table(view, caption="Opto light QC by linkage group",
                     formats={"Lit (s)": "{:.0f}", "Unexplained": "{:.0%}",
                              "From (min)": "{:.0f}", "Contact, no lick (min)": "{:.0f}"},
                     status={"Verdict": rl.tone_of}),
            rl.Paragraph(thresholds, size=rl.SIZE_SMALL, color=rl.MUTED),
        ]
        items = [*self.experiment_notes(), *self._bullets()]
        if items:
            blocks.append(rl.Bullets(items, size=rl.SIZE_SMALL + 0.5))
        for dfm_id in self.dfm_ids():
            if generations.get(dfm_id) == GENERATION_PARADIGM:
                caption = ("Lit time (OnMs) per bin: explained by a trigger-well lick or "
                           "touch within the Hold, or at the onset of a fixed-length "
                           "Activation (grey-blue), unexplained (red), lit while every "
                           "threshold was Off (purple) and not judged by licks — open loop, "
                           "NonFeeding (light grey).  Orange: light of Activations the "
                           "firmware counted with no lick or touch at their onset.")
            else:
                caption = ("Lit time per bin: explained by a trigger-well lick or touch "
                           "within the decay (grey-blue), unexplained (red), lit while every "
                           "threshold was -1 (purple) and not judged by licks — open loop, "
                           "non-feeding activation (light grey).  Orange: time the emulated "
                           "firmware trigger read the trigger wells above threshold with no "
                           "lick or touch.")
            blocks.append(rl.Plot(
                lambda d=dfm_id: self.plot_dfm(d, base_font_size=8.5),
                height=min(8.4, 1.45 * self.plot_rows(dfm_id) + 0.9),
                title=f"Light explained by licks — DFM {dfm_id}",
                caption=caption))
        ## Each Paradigm against its own rule (stream S3).
        if rules:
            blocks += rule_figures.report_blocks(self, rules)
        elif rules_error is not None:
            blocks.append(rl.Callout(f"The rule checks' subsections could not be built: "
                                     f"{rules_error}", tone="failed"))
        return blocks

    def _bullets(self) -> list[str]:
        """:meth:`lines` folded into one bullet per DFM or group."""
        items: list[str] = []
        current: str | None = None
        for line in self.lines():
            stripped = line.strip()
            if not line.startswith(" "):
                if current:
                    items.append(current)
                current = stripped
            elif current is not None:
                sep = " " if current.endswith(":") else "; "
                current += sep + stripped.lstrip("-· ").strip()
        if current:
            items.append(current)
        return items


def _flag_counts(rows: pd.DataFrame, kinds) -> str:
    """How many of *rows*' Linkage Groups raised each flag of *kinds*, in
    *kinds*' order: ``wrong activation length in 2 groups, lockout broken in
    1 group`` — the glance callout's count."""
    counts: dict[str, int] = {}
    for flags in rows["Flags"]:
        for flag in _split_flags(flags):
            if flag in kinds:
                counts[flag] = counts.get(flag, 0) + 1
    ordered = sorted(counts, key=list(kinds).index)
    return ", ".join(f"{flag} in {counts[flag]} group{'s' if counts[flag] != 1 else ''}"
                     for flag in ordered) or "—"

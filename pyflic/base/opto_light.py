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
from .pr_light_qc import VERDICT_FAILED, VERDICT_OK, VERDICT_WARNING

#: ``qc/<QC_SUBDIR>/`` holds the files this module writes.
QC_SUBDIR = "opto"
QC_FILES: dict[str, str] = {
    "opto_light_qc": "opto_light_qc.csv",
    "opto_light_intervals": "opto_light_intervals.csv",
    "opto_light_events": "opto_light_events.csv",
    "opto_program": "opto_program.csv",
}
#: What the light QC adds to every per-chamber summary row of an optogenetic
#: experiment: the chamber's Linkage Groups' flags and their largest
#: unexplained fraction.
ROW_COLUMNS: tuple[str, ...] = ("OptoLightQC", "OptoUnexplainedFraction")

_VERDICT_RANK = {VERDICT_OK: 0, VERDICT_WARNING: 1, VERDICT_FAILED: 2}
_UNEXPLAINED_FLAGS = (oq.UNEXPLAINED, oq.PARTLY_UNEXPLAINED, oq.UNEXPLAINED_NO_PROGRAM)


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
    return oq.DFMInputs(
        dfm_id=int(dfm.id), minutes=minutes,
        samples_per_second=float(dfm.params.samples_per_second),
        lights=_matrix(dfm.lights_df, n, bool), activity=activity,
        feeding=_matrix(dfm.event_df, n, np.int64), raw=signal,
        chamber_of_well=chamber_of_well, origin=recording_origin(raw),
        data_frequency=column("OptoFreq"), data_pulse_width=column("OptoPW"),
        data_dark=column("Dark"))


def _split_flags(text: Any) -> list[str]:
    return [f for f in str(text if text is not None else "").split(", ")
            if f and f.lower() != "nan"]


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
        return oq.decay_profile(dfm_id, minutes, float(dfm.params.samples_per_second),
                                recording_origin(dfm.raw_df), self.program,
                                self.settings())

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

        ``Flags`` lists what fired: *unexplained light*, *light while off* and
        *no light recorded* fail a group; the rest are warnings.  ``Verdict``
        is ``ok``, ``warning`` or ``failed``; ``Excluded`` says a failed
        group's chambers leave the analysis (``exclude_failed_opto_chambers``,
        off by default).  The definitions are in :mod:`pyflic.base.opto_qc`.
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
        return pd.DataFrame(rows, columns=list(oq.OPTO_QC_COLUMNS))

    def interval_table(self) -> pd.DataFrame:
        rows = [r for d in self.dfm_ids() for r in self.result(d).intervals]
        return pd.DataFrame(rows, columns=list(oq.OPTO_INTERVAL_COLUMNS))

    def events_table(self) -> pd.DataFrame:
        rows = [r for d in self.dfm_ids() for r in self.result(d).events]
        return pd.DataFrame(rows, columns=list(oq.OPTO_EVENT_COLUMNS))

    def program_table(self) -> pd.DataFrame:
        program = self.program
        rows = [] if program is None else oq.program_rows(program, self.dfm_ids())
        return pd.DataFrame(rows, columns=list(oq.OPTO_PROGRAM_COLUMNS))

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
        start = program.start.strftime("%Y-%m-%d %H:%M:%S") if program.start else "no start"
        duration = (f"{program.duration_min:g} min" if program.duration_min is not None
                    else "no duration")
        return (f"{name}: {len(program.dfms)} DFM section(s), start {start}, {duration}, "
                f"firmware baseline {'on' if program.baseline else 'off'}")

    def experiment_notes(self) -> list[str]:
        """Notes about the experiment as a whole: the program's parse warnings,
        what the loader derived from it, and a program whose paradigm is not
        the Experiment Type's."""
        exp = self.experiment
        notes: list[str] = []
        program = self.program
        if program is not None:
            shown = program.warnings[:8]
            notes.extend(f"Program.txt: {w}" for w in shown)
            if len(program.warnings) > len(shown):
                notes.append(f"Program.txt: and {len(program.warnings) - len(shown)} "
                             f"more line(s) skipped")
            sections = {d: program.dfms[d] for d in self.dfm_ids() if d in program.dfms}
            if sections:
                paradigms = oq.paradigms_in(op.OptoProgram(
                    path=None, start=None, end=None, duration_min=None, baseline=True,
                    dfms=sections))
                etype = getattr(exp, "experiment_type", None)
                is_pr = getattr(etype, "name", "") == "ProgressiveRatio"
                display = getattr(etype, "display_name", "Custom")
                if op.PROGRESSIVE_RATIO in paradigms and not is_pr:
                    notes.append(f"Program.txt runs a progressive ratio, but the experiment "
                                 f"type is {display}: the breaking point and the paired / "
                                 f"yoked analysis need experiment_type ProgressiveRatio")
                if is_pr and op.PROGRESSIVE_RATIO not in paradigms:
                    listed = ", ".join(sorted(paradigms)) or "no triggered interval"
                    notes.append(f"the experiment type is Progressive Ratio, but Program.txt "
                                 f"runs no progressive-ratio interval ({listed})")
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
            if dfm_flags or result.notes:
                head = f"DFM {dfm_id}"
                if dfm_flags:
                    head += f": {verdict_text(', '.join(dfm_flags))}"
                lines.append(head)
                lines.extend(f"    · {note}" for note in result.notes)
            rows = table[table["DFM"] == dfm_id]
            for _, r in rows.iterrows():
                own = [f for f in _split_flags(r["Flags"]) if f not in dfm_flags]
                reasons = [self._reason(flag, r) for flag in own]
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

    @staticmethod
    def _reason(flag: str, r) -> str:
        if flag in _UNEXPLAINED_FLAGS:
            text = (f"{flag}: {r['UnexplainedSec']:.0f} s of {r['JudgedLitSec']:.0f} s lit "
                    f"({100 * r['UnexplainedFraction']:.0f}%) with no trigger-well lick or "
                    f"touch within the decay")
            if pd.notna(r["UnexplainedOnsetMin"]):
                text += f", from minute {r['UnexplainedOnsetMin']:.0f}"
            if r["LikelyCause"]:
                text += f"; {r['LikelyCause']}"
            return text
        if flag == oq.LIGHT_WHILE_OFF:
            return (f"light while off: lit {r['LitWhileOffSec']:.1f} s while every "
                    f"member's threshold was -1")
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
               f"Settings: {settings.describe()}", ""]
        program = self.program_table()
        if not program.empty:
            show = program[["DFM", "Interval", "DurationMin", "Paradigm", "DecayMs",
                            "Delay", "MaxTimeOn", "FrequencyHz", "AcclimationEvents",
                            "TriggerWells"]].copy()
            show.columns = ["DFM", "Int", "Min", "Paradigm", "Decay", "Delay", "MaxOn",
                            "Hz", "Accl", "Trigger wells"]
            out.append(show.to_string(index=False))
            out.append("")
        table = self.qc_table()
        if table.empty:
            return out + ["(no DFM is covered)", ""]
        show = table[["DFM", "Group", "Wells", "Chambers", "Treatment", "LitSec",
                      "UnexplainedFraction", "UnexplainedOnsetMin",
                      "ContactWithoutActivitySec", "Verdict"]].copy()
        show.columns = ["DFM", "Group", "Wells", "Chambers", "Treatment", "Lit(s)",
                        "Unexpl", "Onset", "Contact(min)", "Verdict"]
        show["Lit(s)"] = show["Lit(s)"].map(lambda v: f"{v:.0f}")
        show["Unexpl"] = show["Unexpl"].map(lambda v: "—" if pd.isna(v) else f"{100 * v:.0f}%")
        show["Onset"] = show["Onset"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}")
        show["Contact(min)"] = show["Contact(min)"].map(
            lambda v: "—" if pd.isna(v) else f"{v / 60.0:.0f}")
        out.append(show.to_string(index=False))
        out.append("(Contact: minutes the emulated firmware trigger saw the signal above "
                   "threshold with no pyflic lick or touch.)")
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
                 base_font_size: float = 10.0, figsize: tuple[float, float] | None = None):
        """Light explained over time (QC) for one DFM, one row per Linkage
        Group that was ever lit or touched.

        The left panel of a row is the group's lit time per bin, stacked:
        explained (a trigger-well lick or touch within the decay), unexplained,
        lit while every threshold was -1, and not judged by licks (open loop,
        non-feeding activation).  The right one, when the program allows the
        Emulated Trigger, is the time the firmware would have read the trigger
        wells above threshold while pyflic saw no lick or touch — the
        drifting-baseline signature.  A group never lit or touched is left
        out unless it was flagged.
        """
        import plotnine as p9

        dfm_id = int(dfm_id)
        if dfm_id not in self.dfm_ids():
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — not covered by the "
                                               f"opto light QC")
        result = self.result(dfm_id)
        binsize = max(float(binsize_min), 1.0)
        emulated = any(pd.notna(g["ContactSec"]) for g in result.groups)
        frames, order = [], []
        for group in result.groups:
            tl = result.timelines.get(group["Group"])
            if tl is None or tl.empty:
                continue
            if not self._drawn(group, tl):
                continue
            head = (f"Group {group['Group']} ({group['Wells']}) — "
                    f"{verdict_text(group['Flags'])}")
            light_panel = f"{head}\nlight (s per bin)"
            contact_panel = f"{head}\nfirmware contact, no lick (s per bin)"
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
                                            "Kind": "firmware contact, no lick"}))
        if not frames:
            return p9.ggplot() + p9.labs(title=f"DFM {dfm_id} — no light to show")
        data = pd.concat(frames, ignore_index=True)
        data["Panel"] = pd.Categorical(data["Panel"], categories=order, ordered=True)
        kinds = ["explained", "unexplained", "lit while off", "not judged by licks",
                 "firmware contact, no lick"]
        data["Kind"] = pd.Categorical(data["Kind"], categories=kinds, ordered=True)
        ncol = 2 if emulated else min(3, len(order))
        nrow = -(-len(order) // ncol)
        if figsize is None:
            figsize = (4.6 * ncol, min(24.0, 1.9 * nrow + 1.0))
        colors = {"explained": "#9DB4C8", "unexplained": "#D62728",
                  "lit while off": "#6A3D9A", "not judged by licks": "#D9D9D9",
                  "firmware contact, no lick": "#E69F00"}
        g = (p9.ggplot(data, p9.aes("Minute", "Seconds", fill="Kind"))
             + p9.geom_col(width=binsize, position="stack")
             ## Every panel from zero, and an empty one a unit tall rather
             ## than centred on nothing.
             + p9.expand_limits(y=[0, 1])
             + p9.facet_wrap("~ Panel", ncol=ncol, scales="free_y")
             + p9.scale_fill_manual(values=colors, drop=False)
             + p9.labs(title=f"DFM {dfm_id} — light explained by trigger-well licks "
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
        """The cover's callout: how many Linkage Groups failed or warned."""
        from . import report_layout as rl

        if not self.active:
            return []
        table = self.qc_table()
        n = len(table)
        failed = table[table["Verdict"] == VERDICT_FAILED]
        warned = table[table["Verdict"] == VERDICT_WARNING]
        name = lambda rows: "; ".join(f"DFM {int(r.DFM)} group {r.Group}"  # noqa: E731
                                      for r in rows.itertuples())
        if not failed.empty:
            excluded = int(failed["Excluded"].sum())
            text = (f"{len(failed)} of {n} linkage groups failed ({name(failed)})"
                    + (f"; {excluded} excluded." if excluded else
                       "; kept and flagged, because exclude_failed_opto_chambers is off."))
            tone = "failed"
        else:
            text = f"Every linkage group's light was explained by its licks ({n} of {n})."
            tone = "ok"
        if not warned.empty:
            text += f"  {len(warned)} warning(s): {name(warned)} — see Quality control."
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
        settings = self.settings()
        blocks: list[Any] = [
            rl.Heading("Optogenetic light QC", level=2),
            rl.Paragraph(
                "The firmware lights a linkage group from its own reading of the group's "
                "trigger wells during the run; pyflic counts licks afterwards, from the "
                "baselined signal.  A lit sample is explained when a trigger well has a "
                "lick or a touch within the interval's light decay, plus "
                f"{settings.tolerance_samples} sample(s), before it.  Unexplained light — "
                "light with no lick behind it — is the sign of a drifting sensor, a "
                "sustained contact or a hardware fault.  Computed over the whole recording."),
        ]
        program_tone = "info" if self.program is not None else "warning"
        blocks.append(rl.Callout(self.program_line(), tone=program_tone, title="Program"))
        program = self.program_table()
        if not program.empty:
            view = pd.DataFrame({
                "DFM": program["DFM"], "Interval": program["Interval"],
                "Minutes": program["DurationMin"], "Paradigm": program["Paradigm"],
                "Decay (ms)": program["DecayMs"], "Delay": program["Delay"],
                "Max time on": program["MaxTimeOn"], "Freq (Hz)": program["FrequencyHz"],
                "Acclimation": program["AcclimationEvents"],
                "Linkage": program["Linkage"], "Trigger wells": program["TriggerWells"],
            })
            blocks.append(rl.Table(view, caption="The program, as the MCU echoed it",
                                   formats={"Minutes": "{:g}"}))
        table = self.qc_table()
        view = pd.DataFrame({
            "DFM": table["DFM"], "Group": table["Group"], "Wells": table["Wells"],
            "Chambers": table["Chambers"], "Treatment": table["Treatment"],
            "Lit (s)": table["LitSec"],
            "Unexplained": table["UnexplainedFraction"],
            "From (min)": table["UnexplainedOnsetMin"],
            "Contact, no lick (min)": table["ContactWithoutActivitySec"] / 60.0,
            "Verdict": [("excluded" if ex else verdict_text(f))
                        for f, ex in zip(table["Flags"], table["Excluded"])],
        })
        blocks += [
            rl.Table(view, caption="Opto light QC by linkage group",
                     formats={"Lit (s)": "{:.0f}", "Unexplained": "{:.0%}",
                              "From (min)": "{:.0f}", "Contact, no lick (min)": "{:.0f}"},
                     status={"Verdict": rl.tone_of}),
            rl.Paragraph(f"Thresholds: {settings.describe()}.", size=rl.SIZE_SMALL,
                         color=rl.MUTED),
        ]
        items = [*self.experiment_notes(), *self._bullets()]
        if items:
            blocks.append(rl.Bullets(items, size=rl.SIZE_SMALL + 0.5))
        for dfm_id in self.dfm_ids():
            blocks.append(rl.Plot(
                lambda d=dfm_id: self.plot_dfm(d, base_font_size=8.5),
                height=min(8.4, 1.45 * self.plot_rows(dfm_id) + 0.9),
                title=f"Light explained by licks — DFM {dfm_id}",
                caption="Lit time per bin: explained by a trigger-well lick or touch "
                        "within the decay (grey-blue), unexplained (red), lit while every "
                        "threshold was -1 (purple) and not judged by licks — open loop, "
                        "non-feeding activation (light grey).  Orange: time the emulated "
                        "firmware trigger read the trigger wells above threshold with no "
                        "lick or touch."))
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

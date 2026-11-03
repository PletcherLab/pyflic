"""The Program timeline: what a DFM's Opto Program ran, and what each Well did.

One figure per DFM (:func:`plot_dfm`), in matplotlib:

* a top band of **Program Intervals**, coloured by Paradigm type and labelled
  by Paradigm name, with the **Cycle** when the program repeats;
* below it one lane per Well: its Phase ``A`` (Acclimation) shaded, its
  Activation onsets as ticks, and the share of each bin its LED was on as a
  heat strip.  The lane label carries the chamber's **Role**
  (:mod:`pyflic.base.opto_roles`) and, in bold for a Trigger Well, the
  thresholds the program gives it (:func:`well_thresholds`).  The strip
  runs to a whole bin lit when the program lights a Well regardless (open
  loop), and otherwise to the DFM's brightest bin, so the few percent of a
  bin that feeding-driven light fills still shows.

Generation 3 reads the firmware's own record (``dfm.paradigm``): the band is
``ParadigmState.interval_runs`` and the light is ``OnMs``; each Well's lane
also shades the Program Intervals in which ``Program.txt`` sets it ``On``
(hatched) or ``Off`` (grey), and each restart of the DFM is a marked band
across the figure (decision 15 of ``docs/paradigm-analyses-plan.md``).
Generations 1 and 2 have no such record; with a ``Program.txt`` section the
band is the program's clock schedule laid over the recording (approximate,
as the light QC uses it), the strip is ``OptoCol1``, and a Trigger Well's
ticks are its light coming on.  Without
either there is nothing to draw, and :func:`plot_dfm` returns a placeholder
figure.

:func:`well_table` is the same DFM as a table at the end of the loaded range,
and :func:`activations_table` its Activations, for the QC Viewer's Program
sub-tab; :func:`write` saves the figures to
``qc/program/program_timeline_dfm<id>.png``.  Terms are ``CONTEXT.md``'s.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import opto_program as op
from . import opto_roles
from .paradigm_state import GENERATION_DESCRIPTIONS, GENERATION_PARADIGM, RECORD_MS

#: ``qc/<QC_SUBDIR>/`` holds the timelines.
QC_SUBDIR = "program"
#: The per-well table's columns (:func:`well_table`).
WELL_TABLE_COLUMNS: tuple[str, ...] = (
    "Well", "Chamber", "Role", "Paradigm", "Phase", "AcclimationEndMin", "Activations",
    "Requirement", "Progress", "LitSec")

#: Paradigm type (PyMCU spelling) -> band colour, Okabe-Ito in a fixed order
#: so a type keeps its colour on every DFM and in every report.
TYPE_COLORS: dict[str, str] = {
    "OpenLoop": "#E69F00", "ClosedLoop": "#56B4E9", "FixedInterval": "#009E73",
    "NonFeeding": "#CC79A7", "ProgressiveRatio": "#0072B2", "FixedRatio": "#D55E00",
}
_BASELINE = "Baseline"
_UNKNOWN_INSTRUCTION = "Unknown instruction"
_UNKNOWN_TYPE = "Type unknown (not in Program.txt)"
#: PyMCU's own Paradigm: every well Off, nothing lit.  Known without a
#: Program.txt, so never "type unknown".
_IDLE = "Idle (no light)"
_KIND_COLORS = {_BASELINE: "#DDDDDD", _UNKNOWN_INSTRUCTION: "#FFFFFF", _UNKNOWN_TYPE: "#999999",
                _IDLE: "#EEEEEE"}
_ACCLIMATION_COLOR = "#F2C879"
_TICK_COLOR = "#222222"
_HEAT_CMAP = "Blues"
_REPEATING = ("repeating", "circadian")
#: A Well's ``On`` and ``Off`` thresholds in its lane (generation 3): On
#: hatched in the OpenLoop band colour, as it is lit regardless; Off grey.
_ON_COLOR = TYPE_COLORS["OpenLoop"]
_OFF_COLOR = "#DADADA"
#: A DFM restart's band (decision 15).
_RESTART_COLOR = "#B42318"
#: The QC Viewer's *Activations* table: the ledger's rows of
#: ``opto_light_events.csv`` for one DFM (:func:`activations_table`).
ACTIVATION_TABLE_COLUMNS: tuple[str, ...] = (
    "Well", "OnsetMin", "Family", "Phase", "K", "LitMs", "EndedBy", "RequirementMet",
    "Overlap")


@dataclass(frozen=True)
class _Run:
    """One stretch of the band, in recording minutes."""

    start: float
    end: float
    label: str
    kind: str          # a TYPE_COLORS key or one of the _KIND_COLORS keys


# ---------------------------------------------------------------------------
# Which DFMs have a timeline
# ---------------------------------------------------------------------------

def _section(exp, dfm_id: int) -> op.DFMProgram | None:
    program = getattr(exp, "opto_program", None)
    return program.section(int(dfm_id)) if program is not None else None


def has_timeline(exp, dfm_id: int) -> bool:
    """Whether *dfm_id* has something to draw: a Paradigm state (generation
    3) or a ``Program.txt`` section."""
    try:
        dfm = exp.dfms[int(dfm_id)]
        if getattr(dfm, "generation", None) == GENERATION_PARADIGM \
                and getattr(dfm, "paradigm", None) is not None:
            return True
        return _section(exp, dfm_id) is not None
    except Exception:  # noqa: BLE001 - a stand-in experiment has no timeline
        return False


def timeline_dfms(exp) -> list[int]:
    """The DFMs of *exp* that :func:`has_timeline`, in order, and only for an
    optogenetic experiment.  A recording whose light never came on (every
    well Off, as a recording without a Program is) has nothing for the
    timeline to show, so the report and ``qc/`` leave it out, as the Hub's
    button and the script action already do."""
    try:
        if getattr(exp, "is_optogenetic", False) is not True:
            return []
        ids = sorted(int(d) for d in exp.dfms)
    except Exception:  # noqa: BLE001
        return []
    return [d for d in ids if has_timeline(exp, d)]


def timeline_path(qc_dir: str | Path, dfm_id: int) -> Path:
    """``qc/program/program_timeline_dfm<id>.png``."""
    return Path(qc_dir) / QC_SUBDIR / f"program_timeline_dfm{int(dfm_id)}.png"


# ---------------------------------------------------------------------------
# The band
# ---------------------------------------------------------------------------

def _record_minutes(dfm) -> float:
    return 1.0 / (60.0 * float(dfm.params.samples_per_second))


def _paradigm_label(paradigm: op.Paradigm) -> str:
    """A Paradigm's name, or its type for a name PyMCU's legacy conversion
    made up from old settings (``ClosedLoop-40Hz-8ms-Hold500-MaxOn0``)."""
    return paradigm.type_name if paradigm.generated else paradigm.name


def _state_runs(dfm, program: op.OptoProgram | None, section: op.DFMProgram | None) -> list[_Run]:
    """The band from the rows' own ``Interval``, ``Cycle`` and ``Paradigm``."""
    state = dfm.paradigm
    n = len(state)
    if not n:
        return []
    period = _record_minutes(dfm)
    runs = state.interval_runs()
    cycles = {r.cycle for r in runs if r.kind == "program" and r.cycle is not None}
    repeating = (section is not None and section.program_type.strip().lower() in _REPEATING) \
        or any(c > 1 for c in cycles)
    paradigms = program.paradigms if program is not None else {}
    out: list[_Run] = []
    for run in runs:
        a = float(state.minutes[run.i0])
        b = float(state.minutes[run.i1]) if run.i1 < n else float(state.minutes[-1]) + period
        if run.kind == "baseline":
            out.append(_Run(a, b, _BASELINE, _BASELINE))
            continue
        if run.kind == "unknown":
            out.append(_Run(a, b, run.paradigm or "?", _UNKNOWN_INSTRUCTION))
            continue
        name = run.paradigm or "?"
        found = paradigms.get(name)
        if found is None and name == "Idle":
            out.append(_Run(a, b, name, _IDLE))
            continue
        kind = found.type_name if found is not None else _UNKNOWN_TYPE
        label = name + (f" · C{run.cycle}" if repeating and run.cycle is not None else "")
        out.append(_Run(a, b, label, kind if kind in TYPE_COLORS else _UNKNOWN_TYPE))
    return out


def _schedule(dfm, program: op.OptoProgram, section: op.DFMProgram):
    """``[(start minute, end minute, ScheduledInterval)]``: the program's
    clock schedule on the recording's minutes, as the light QC lays it out on
    generation 1 (``opto_qc._program_windows``)."""
    from .opto_light import recording_origin

    minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").to_numpy(dtype=float)
    origin = recording_origin(dfm.raw_df) or program.start or datetime(2000, 1, 1)
    anchor = program.start or origin
    last = float(np.nanmax(minutes)) if minutes.size else 0.0
    end = program.resolved_end() or origin + timedelta(
        minutes=last + _record_minutes(dfm))
    return [((s.start - origin).total_seconds() / 60.0, (s.end - origin).total_seconds() / 60.0, s)
            for s in section.schedule(anchor, end)]


def _schedule_runs(dfm, program: op.OptoProgram, section: op.DFMProgram) -> list[_Run]:
    repeating = section.program_type.strip().lower() in _REPEATING
    out = []
    for a, b, sched in _schedule(dfm, program, section):
        paradigm = sched.interval.paradigm
        label = _paradigm_label(paradigm) + (f" · C{sched.occurrence}" if repeating else "")
        kind = paradigm.type_name if paradigm.type_name in TYPE_COLORS else _UNKNOWN_TYPE
        out.append(_Run(a, b, label, kind))
    return out


# ---------------------------------------------------------------------------
# The lanes
# ---------------------------------------------------------------------------

def _well_states(dfm, section: op.DFMProgram | None, minutes: np.ndarray,
                 period: float) -> list[tuple[int, float, float, str]]:
    """``[(well, start, end, "On" | "Off")]`` on a generation-3 recording:
    each run of rows under a Program Interval, joined to its printed line
    in ``Program.txt``, gives each Well set ``On`` or ``Off`` there a stretch.
    A run beyond the printed lines (their thresholds unknown) gives none."""
    state = dfm.paradigm
    n = len(state)
    if section is None or not n:
        return []
    out: list[tuple[int, float, float, str]] = []
    for run in state.interval_runs():
        if run.kind != "program" or run.program_interval is None:
            continue
        k = int(run.program_interval)
        if not 1 <= k <= len(section.intervals):
            continue
        line = section.intervals[k - 1]
        a = float(minutes[run.i0])
        b = float(minutes[run.i1]) if run.i1 < n else float(minutes[-1]) + period
        for w in range(1, op.N_WELLS + 1):
            threshold = line.threshold(w)
            if threshold == op.ON:
                out.append((w, a, b, "On"))
            elif threshold == op.OFF:
                out.append((w, a, b, "Off"))
    return out


def _restart_spans(dfm, minutes: np.ndarray, period: float) -> list[tuple[float, float]]:
    """``[(start, end)]`` in recording minutes of each restart the rows show
    (:meth:`DFM.restarts <pyflic.base.dfm.DFM.restarts>`, decision 15): from
    the end of the last record before the gap to the first row under the
    instruction again.  A restart only ``Messages.csv`` names has no rows to
    place it by and is left out."""
    try:
        found = dfm.restarts() or []
    except Exception:  # noqa: BLE001 - a figure must not fail on a message file
        return []
    n = minutes.size
    out: list[tuple[float, float]] = []
    for restart in found:
        row, resume = restart.get("Row"), restart.get("ResumeRow")
        if row is None or not 0 <= int(row) < n:
            continue
        row = int(row)
        a = float(minutes[row - 1]) + period if row > 0 else float(minutes[row])
        b = (float(minutes[int(resume)]) if resume is not None and int(resume) < n
             else float(minutes[-1]) + period)
        out.append((a, max(b, a + period)))
    return out


def _bin_edges(minutes: np.ndarray, binsize_min: float, period: float) -> np.ndarray:
    binsize = max(float(binsize_min), period)
    lo = float(np.floor(np.nanmin(minutes) / binsize) * binsize)
    hi = float(np.nanmax(minutes)) + period
    edges = np.unique(np.minimum(np.arange(lo, hi + binsize, binsize), hi))
    return edges if edges.size > 1 else np.array([lo, hi])


def _lit_share(minutes: np.ndarray, lit: np.ndarray, record: np.ndarray,
               edges: np.ndarray) -> np.ndarray:
    """``(bins, 12)``: the share of each bin each Well's LED was on, *lit*
    the per-row on-time and *record* the per-row time the row covers."""
    which = np.clip(np.searchsorted(edges, minutes, side="right") - 1, 0, edges.size - 2)
    bins = edges.size - 1
    on = np.zeros((bins, lit.shape[1]))
    covered = np.zeros(bins)
    np.add.at(on, which, lit)
    np.add.at(covered, which, record)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = on / covered[:, None]
    return np.clip(np.nan_to_num(share, nan=0.0), 0.0, 1.0)


def _roles(dfm, section: op.DFMProgram | None) -> tuple[dict[int, int], dict[int, str]]:
    chamber_of = opto_roles.dfm_chamber_map(dfm)
    roles = opto_roles.chamber_roles(opto_roles.dfm_linkage_view(dfm, section), chamber_of)
    return chamber_of, roles


def well_thresholds(section: op.DFMProgram | None) -> tuple[dict[int, tuple[int, ...]], set[int]]:
    """``(trigger wells → their thresholds, open wells)`` from a program
    section: the levels a well has under a Paradigm that is not OpenLoop,
    where its own contact decides (as :meth:`DFMProgram.trigger_wells
    <pyflic.base.opto_program.DFMProgram.trigger_wells>` counts them), each
    level once in order of first use; and the wells lit regardless in some
    Interval, ``On`` or a level under OpenLoop, that are never a trigger.
    Both empty without a section."""
    if section is None:
        return {}, set()
    levels: dict[int, list[int]] = {}
    open_: set[int] = set()
    for interval in section.intervals:
        if interval.paradigm.type == op.OPEN_LOOP:
            open_ |= set(interval.trigger_wells())
        else:
            for well in interval.trigger_wells():
                found = levels.setdefault(well, [])
                if interval.threshold(well) not in found:
                    found.append(interval.threshold(well))
        open_ |= set(interval.open_wells())
    return {w: tuple(v) for w, v in levels.items()}, open_ - set(levels)


def _scale_top(share: np.ndarray, full: bool) -> float:
    """The light strip's top: the whole bin when *full* (the program lights
    a well regardless, so a lane lit throughout is what to see), else the
    DFM's brightest bin rounded up to 1, 2 or 5 in its decade, so light that
    fills a few percent of a bin, as feeding-driven light does, still shows."""
    peak = float(np.nanmax(share)) if share.size else 0.0
    if full or not peak > 0.0:
        return 1.0
    decade = 10.0 ** np.floor(np.log10(peak))
    for step in (1.0, 2.0, 5.0, 10.0):
        if step * decade >= peak * (1 - 1e-9):
            return min(1.0, step * decade)
    return 1.0


def _onsets(lit: np.ndarray) -> np.ndarray:
    """The rows a well's light came on: each rise of *lit* (one well's column)."""
    on = np.asarray(lit, dtype=float) > 0
    return np.flatnonzero(on & ~np.r_[False, on[:-1]])


# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

def _threshold_label(well: int, triggers: dict[int, tuple[int, ...]],
                     open_wells: set[int]) -> str:
    """The lane label's last part: ``trigger 20`` for a trigger well, with
    every level it has (``trigger 20, 40``) or none when unknown; ``On``
    for a well lit regardless; nothing for a well the program keeps Off."""
    if well in triggers:
        levels = triggers[well]
        return "  trigger" + (" " + ", ".join(str(v) for v in levels) if levels else "")
    return "  On" if well in open_wells else ""


def _placeholder(text: str):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8.0, 1.4))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=10, color="#444444", wrap=True)
    return fig


def _ink_for(color: str) -> str:
    """Black or white text, whichever reads on *color*."""
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(color)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "#111111" if luminance > 0.45 else "#FFFFFF"


def plot_dfm(exp, dfm_id: int, *, binsize_min: float = 10.0):
    """The Program timeline of *dfm_id* as a matplotlib ``Figure`` (built
    without pyplot, so it is safe in a worker thread).

    *binsize_min* is the width of the light heat strip's bins.  A DFM with
    neither a Paradigm state nor a program section gets a placeholder figure
    saying so.
    """
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import PercentFormatter

    dfm_id = int(dfm_id)
    dfm = exp.dfms[dfm_id]
    program = getattr(exp, "opto_program", None)
    section = _section(exp, dfm_id)
    state = dfm.paradigm if dfm.generation == GENERATION_PARADIGM else None
    if state is None and section is None:
        why = ("there is no Program.txt" if program is None
               else "Program.txt has no section for this DFM")
        return _placeholder(
            f"DFM {dfm_id}: no Program timeline. Generation {dfm.generation} "
            f"({GENERATION_DESCRIPTIONS.get(dfm.generation, 'unknown')}) records no Paradigm "
            f"state, and {why}.")
    minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").to_numpy(dtype=float)
    if not minutes.size or not np.isfinite(minutes).any():
        return _placeholder(f"DFM {dfm_id}: no rows loaded.")
    period = _record_minutes(dfm)

    if state is not None:
        runs = _state_runs(dfm, program, section)
        source = ("Program Intervals as the DFM recorded them (generation 3); light from OnMs"
                  + ("" if section is not None else "; no Program.txt section for this DFM"))
        lit = state.on_ms.astype(float)
        record = np.full(minutes.size, float(RECORD_MS))
    else:
        runs = _schedule_runs(dfm, program, section)
        source = (f"Program Intervals from Program.txt's clock schedule (generation "
                  f"{dfm.generation}, approximate); light from OptoCol1, once a record; "
                  f"ticks are light onsets")
        lights = dfm.lights_df
        lit = (lights[[f"W{w}" for w in range(1, 13)]].to_numpy(dtype=float)
               if lights is not None and len(lights) else np.zeros((minutes.size, 12)))
        record = np.ones(minutes.size)
    edges = _bin_edges(minutes, binsize_min, period)
    share = _lit_share(minutes, lit, record, edges)
    chamber_of, roles = _roles(dfm, section)
    ## What each Well's threshold makes it: a trigger well at its level(s),
    ## or lit regardless.  Without a section a generation-3 trigger well is
    ## one that counted Activations, its level unknown.
    triggers, open_wells = well_thresholds(section)
    if section is None and state is not None:
        triggers = {int(w): () for w in state.wells_with_activations()}
    top = _scale_top(share, full=bool(open_wells))
    ## Generation 3 only (generations 1 and 2 keep their figure): each Well's
    ## On and Off stretches, and the DFM's restarts.
    states = _well_states(dfm, section, minutes, period) if state is not None else []
    restarts = _restart_spans(dfm, minutes, period) if state is not None else []

    x0, x1 = float(edges[0]), float(edges[-1])
    fig = Figure(figsize=(11.0, 6.4), layout="constrained")
    band_ax, lane_ax = fig.subplots(2, 1, sharex=True,
                                    gridspec_kw={"height_ratios": [1.0, 7.5]})
    fig.suptitle(f"DFM {dfm_id} — Program timeline", fontsize=12, x=0.01, ha="left")
    band_ax.set_title(source, fontsize=8.5, loc="left", color="#555555")

    ## The band.
    span = max(x1 - x0, period)
    kinds: list[str] = []
    for run in runs:
        a, b = max(run.start, x0), min(run.end, x1)
        if b <= a:
            continue
        color = TYPE_COLORS.get(run.kind) or _KIND_COLORS[run.kind]
        band_ax.add_patch(Rectangle((a, 0.08), b - a, 0.84, facecolor=color, edgecolor="white",
                                    linewidth=1.0, hatch="//" if run.kind == _UNKNOWN_INSTRUCTION
                                    else None))
        if run.kind not in kinds:
            kinds.append(run.kind)
        if (b - a) / span >= 0.045:
            band_ax.text((a + b) / 2.0, 0.5, run.label, ha="center", va="center", fontsize=7.5,
                         color=_ink_for(color), clip_on=True)
    band_ax.set_ylim(0, 1)
    band_ax.set_yticks([])
    band_ax.set_ylabel("Interval", rotation=0, ha="right", va="center", fontsize=8.5)
    for side in ("top", "right", "left"):
        band_ax.spines[side].set_visible(False)

    ## The lanes, Well 1 at the top.
    norm = Normalize(vmin=0.0, vmax=top)
    labels, ticks, onset_ticks = [], [], False
    for well, a, b, kind in states:
        a, b = max(a, x0), min(b, x1)
        if b <= a:
            continue
        y = 12 - well
        if kind == "On":
            lane_ax.add_patch(Rectangle((a, y + 0.02), b - a, 0.40, facecolor="white",
                                        edgecolor=_ON_COLOR, hatch="////", linewidth=0.4))
        else:
            lane_ax.add_patch(Rectangle((a, y + 0.02), b - a, 0.40, facecolor=_OFF_COLOR,
                                        edgecolor="none"))
    for w in range(1, 13):
        y = 12 - w
        ticks.append(y)
        chamber = chamber_of.get(w)
        role = roles.get(chamber, "") if chamber is not None else ""
        labels.append(f"W{w}" + (f"  ch {chamber}" if chamber is not None else "")
                      + (f"  {role}" if role else "") + _threshold_label(w, triggers, open_wells))
        heat = (y - 0.42, y - 0.02)
        if state is not None:
            for i0, i1 in state.acclimation_runs(w):
                a = float(minutes[i0])
                b = float(minutes[i1]) if i1 < minutes.size else float(minutes[-1]) + period
                lane_ax.add_patch(Rectangle((a, y + 0.02), b - a, 0.40, facecolor=_ACCLIMATION_COLOR,
                                            edgecolor="none"))
            onsets = np.unique(state.activation_onsets(w))
        else:
            ## No record of the firmware's Activations: a trigger well's own
            ## light coming on is the nearest thing, and under a ratio or
            ## fixed-interval Paradigm each onset is one Activation.
            onsets = _onsets(lit[:, w - 1]) if w in triggers else np.zeros(0, dtype=int)
            onset_ticks = onset_ticks or bool(onsets.size)
        if onsets.size:
            lane_ax.vlines(minutes[onsets], y + 0.06, y + 0.38,
                           color=_TICK_COLOR, linewidth=0.6)
        mesh = lane_ax.pcolormesh(edges, np.array(heat), share[:, w - 1][None, :],
                                  cmap=_HEAT_CMAP, norm=norm, shading="flat")
    ## Chamber boundaries.
    for w in range(1, 12):
        if chamber_of.get(w) != chamber_of.get(w + 1):
            lane_ax.axhline(12 - w - 0.5, color="#CCCCCC", linewidth=0.6)
    lane_ax.set_yticks(ticks)
    lane_ax.set_yticklabels(labels, fontsize=8)
    for w, label in enumerate(lane_ax.get_yticklabels(), start=1):
        if w in triggers:
            label.set_fontweight("bold")
    lane_ax.set_ylim(-0.5, 11.5)
    lane_ax.set_xlim(x0, x1)
    lane_ax.set_xlabel("Minutes")
    lane_ax.tick_params(axis="y", length=0)
    for side in ("top", "right"):
        lane_ax.spines[side].set_visible(False)
    bar = fig.colorbar(mesh, ax=lane_ax, location="right", shrink=0.5, pad=0.01,
                       format=PercentFormatter(xmax=1.0, decimals=None))
    bar.set_label(f"% of each {float(binsize_min):g}-min bin lit"
                  + ("" if top >= 1.0 else ", scaled to this DFM"), fontsize=8)
    bar.ax.tick_params(labelsize=7)

    ## Each restart: a band over the Intervals and the lanes, at least a
    ## line wide, labelled in the band (decision 15).
    for a, b in restarts:
        b = max(b, a + 0.004 * span)
        for ax in (band_ax, lane_ax):
            ax.axvspan(a, b, facecolor=_RESTART_COLOR, alpha=0.22, edgecolor=_RESTART_COLOR,
                       linewidth=0.8, zorder=3)
        band_ax.text(b, 1.0, " restart", ha="left", va="bottom", fontsize=7.5,
                     color=_RESTART_COLOR, clip_on=False)

    handles: list[Any] = [Patch(facecolor=TYPE_COLORS.get(k) or _KIND_COLORS[k],
                                edgecolor="#999999", linewidth=0.5,
                                hatch="//" if k == _UNKNOWN_INSTRUCTION else None, label=k)
                          for k in kinds]
    if state is not None:
        handles += [Patch(facecolor=_ACCLIMATION_COLOR, label="Phase A (Acclimation)"),
                    Line2D([], [], color=_TICK_COLOR, linewidth=1.0, label="Activation")]
        if any(kind == "On" for _w, _a, _b, kind in states):
            handles.append(Patch(facecolor="white", edgecolor=_ON_COLOR, hatch="////",
                                 linewidth=0.4, label="Well On"))
        if any(kind == "Off" for _w, _a, _b, kind in states):
            handles.append(Patch(facecolor=_OFF_COLOR, label="Well Off"))
        if restarts:
            handles.append(Patch(facecolor=_RESTART_COLOR, alpha=0.4, label="DFM restart"))
    elif onset_ticks:
        handles.append(Line2D([], [], color=_TICK_COLOR, linewidth=1.0,
                              label="Light onset at a trigger well"))
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncols=min(len(handles), 6),
                   fontsize=8, frameon=False)
    return fig


# ---------------------------------------------------------------------------
# The per-well table
# ---------------------------------------------------------------------------

def well_table(exp, dfm_id: int) -> pd.DataFrame:
    """Each Well of *dfm_id* at the end of the loaded range: its chamber and
    Role, the Paradigm in force, its Phase, the minute its Acclimation ended,
    its Activations so far, its Requirement and Progress, and the seconds its
    LED was on over the loaded range (:data:`WELL_TABLE_COLUMNS`).

    Read from the loaded experiment, so a parameter recompute that replaces
    the DFM is reflected.  The Paradigm-state columns are ``NA`` on
    generations 1 and 2, whose Paradigm is the program's scheduled one.
    """
    dfm_id = int(dfm_id)
    dfm = exp.dfms[dfm_id]
    section = _section(exp, dfm_id)
    state = dfm.paradigm if dfm.generation == GENERATION_PARADIGM else None
    chamber_of, roles = _roles(dfm, section)
    lit = dfm.light_seconds()
    rows = []
    if state is not None and len(state):
        last = len(state) - 1
        name = state.paradigm[last]
        paradigm = None if pd.isna(name) else str(name)
        for w in range(1, 13):
            end = state.acclimation_end(w)
            requirement = state.requirement[last, w - 1]
            progress = state.progress[last, w - 1]
            rows.append({
                "Phase": "A" if state.phase_a[last, w - 1] else "M",
                "AcclimationEndMin": np.nan if end is None else float(state.minutes[end]),
                "Activations": int(state.activations[last, w - 1]),
                "Requirement": None if np.isnan(requirement) else int(requirement),
                "Progress": None if np.isnan(progress) else int(progress),
                "Paradigm": paradigm,
            })
    else:
        paradigm = None
        minutes = pd.to_numeric(dfm.raw_df["Minutes"], errors="coerce").to_numpy(dtype=float)
        program = getattr(exp, "opto_program", None)
        if section is not None and program is not None and minutes.size:
            ## The last scheduled Interval that started by the last row.
            at = float(np.nanmax(minutes))
            for a, _b, sched in _schedule(dfm, program, section):
                if a <= at:
                    paradigm = _paradigm_label(sched.interval.paradigm)
        rows = [{"Phase": None, "AcclimationEndMin": np.nan, "Activations": None,
                 "Requirement": None, "Progress": None, "Paradigm": paradigm}
                for _ in range(12)]
    for w, row in enumerate(rows, start=1):
        chamber = chamber_of.get(w)
        row.update({"Well": w, "Chamber": chamber,
                    "Role": roles.get(chamber, "") if chamber is not None else "",
                    "LitSec": float(lit[w - 1]) if len(lit) == 12 else np.nan})
    frame = pd.DataFrame(rows, columns=list(WELL_TABLE_COLUMNS))
    for column in ("Chamber", "Activations", "Requirement", "Progress"):
        frame[column] = pd.array(frame[column].tolist(), dtype="Int64")
    return frame


def activations_table(exp, dfm_id: int) -> pd.DataFrame:
    """Every Activation of *dfm_id*'s Trigger Wells, one row each: the
    ledger's rows of the light QC's ``opto_light_events.csv``
    (:data:`ACTIVATION_TABLE_COLUMNS`: its Well, onset minute, Paradigm
    Family, Phase, Main number K, lit time, what ended it, the Requirement it
    met and whether other light merged with it), for the QC Viewer's Program
    sub-tab.  Empty for a DFM the light QC does not cover or that records no
    Paradigm state (generations 1 and 2)."""
    dfm_id = int(dfm_id)
    empty = pd.DataFrame(columns=list(ACTIVATION_TABLE_COLUMNS))
    opto = getattr(exp, "opto", None)
    dfm = exp.dfms[dfm_id]
    if opto is None or getattr(dfm, "generation", None) != GENERATION_PARADIGM \
            or dfm_id not in opto.dfm_ids():
        return empty
    from .opto_qc import EVENT_SOURCE_ACTIVATIONS

    events = opto.events_table()
    rows = events[(events["DFM"] == dfm_id) & (events["Source"] == EVENT_SOURCE_ACTIVATIONS)]
    if rows.empty:
        return empty
    return rows.sort_values(["OnsetMin", "Well"], kind="stable")[
        list(ACTIVATION_TABLE_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def write(exp, out_dir: str | Path | None = None, *, dfm_ids=None, binsize_min: float = 10.0,
          dpi: int = 150) -> dict[str, Path]:
    """Save ``program_timeline_dfm<id>.png`` for each of *dfm_ids* (default:
    every DFM with a timeline) to *out_dir* (default ``qc/program/``) and
    return ``{"program_timeline_dfm<id>": path}``."""
    if out_dir is None:
        qc_dir = getattr(exp, "qc_dir", None)
        if qc_dir is None:
            raise ValueError("experiment_dir must be set to write the Program timeline.")
        out_dir = Path(qc_dir) / QC_SUBDIR
    out = Path(out_dir).expanduser().resolve()
    ids = timeline_dfms(exp) if dfm_ids is None else [int(d) for d in dfm_ids]
    written: dict[str, Path] = {}
    for dfm_id in ids:
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"program_timeline_dfm{dfm_id}.png"
        plot_dfm(exp, dfm_id, binsize_min=binsize_min).savefig(
            path, dpi=dpi, facecolor="white")
        written[f"program_timeline_dfm{dfm_id}"] = path
    return written

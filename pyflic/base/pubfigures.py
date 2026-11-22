"""Publication Figures: plotnine-rendered, journal-ready vector figures.

A separate rendering path from the matplotlib figures embedded in the PDF
report: the two share the same summarized, exclusion-filtered data — never
rendering code.  A figure is defined by a per-plot :class:`PlotSpec` (content)
plus a named, reusable :class:`PlotStyle` (look).  Both persist in
``<project>/plot_specs.yaml``, written by the Plot Editor; saved figures land in
``<project>/figures/``.

pyflic has **three** spec families where PyTrackingAnalysis has one:

* ``faceted_<metric>`` — x = Treatment, one panel per Facet, jittered points
  with a mean overlay.  Expressible only because a time window is now a column
  (ADR-0008).  The light metrics (``faceted_activations``,
  ``faceted_litsec``, …; :data:`LIGHT_METRICS`) are the same figure drawn
  from the stacked light table, offered when a Project has one.
* ``timecourse_<metric>`` — x = time bin, one line per Treatment with an SEM
  ribbon.  The figure that most distinguishes FLIC data, and the reason the
  Editor carries two preview forms.
* ``scatter_pr_breaking_point`` — Progressive Ratio only: one point per chamber
  group, Breaking Point against Test-phase minutes (``BreakMin``).
* ``dot_pr_breaking_point`` — Progressive Ratio only: Breaking Point against
  Treatment (no Phase/Role facets).

All share the ``styles:`` block, so a Project has one look.

SVG output uses ``svg.fonttype='none'`` so labels arrive in Illustrator as live,
editable text; PDF embeds TrueType (fonttype 42) for the same reason.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache

import matplotlib
import pandas as pd
import yaml

SPECS_FILENAME = "plot_specs.yaml"
FIGURES_DIRNAME = "figures"

#: Ordered fallback palette for treatments a style does not name explicitly.
DEFAULT_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
                   "#7c3aed", "#0891b2", "#64748b", "#be185d"]

#: Progressive Ratio Role columns left-to-right (yoked control, then paired).
ROLE_ORDER = ("yoked", "paired")


def order_roles(roles) -> list[str]:
    """Stable Role order for facet columns: yoked, then paired, then any other."""
    seen = list(dict.fromkeys(str(r) for r in roles
                              if r and str(r).lower() != "nan"))
    known = [r for r in ROLE_ORDER if r in seen]
    other = [r for r in seen if r not in ROLE_ORDER]
    return known + other


def role_display_label(role: str) -> str:
    """Strip label for a Role: ``yoked``/``paired`` → ``Yoked``/``Paired``."""
    text = str(role).strip()
    if text.lower() in ROLE_ORDER:
        return text.lower().capitalize()
    return text

_THEMES = ("classic", "bw", "minimal")
_MEAN_STYLES = ("point+sem", "bar+sem", "point+95ci", "bar+95ci")
_GEOMS = ("dots", "box", "box+dots")
_STRIP_STYLES = ("plain", "boxed")

FAMILY_FACETED = "faceted"
FAMILY_TIMECOURSE = "timecourse"
FAMILY_SCATTER = "scatter"

#: Every plot id the Editor and the report set can name.  ``metric`` is
#: resolved through ``analytics._resolve_metric_col``, so a two-well summary's
#: ``LicksA``/``LicksB`` answer to ``Licks`` without a per-layout table here.
PLOT_TYPES: dict[str, dict] = {
    "faceted_pi": {
        "family": FAMILY_FACETED, "metric": "PI",
        "y_label": "Preference index (well A)",
        "y_limits": (-1.0, 1.0), "ref_line": 0.0,
        "display": "Preference index (faceted)", "layout": "two_well",
    },
    "faceted_event_pi": {
        "family": FAMILY_FACETED, "metric": "EventPI",
        "y_label": "Event preference index (well A)",
        "y_limits": (-1.0, 1.0), "ref_line": 0.0,
        "display": "Event preference index (faceted)", "layout": "two_well",
    },
    "faceted_licks": {
        "family": FAMILY_FACETED, "metric": "Licks", "y_label": "Licks",
        "y_limits": None, "ref_line": None, "free_y": True,
        "display": "Licks (faceted)",
    },
    "faceted_events": {
        "family": FAMILY_FACETED, "metric": "Events", "y_label": "Feeding events",
        "y_limits": None, "ref_line": None, "free_y": True,
        "display": "Events (faceted)",
    },
    "faceted_medduration": {
        "family": FAMILY_FACETED, "metric": "MedDuration",
        "y_label": "Median bout duration (s)",
        "y_limits": None, "ref_line": None, "free_y": True,
        "display": "Median duration (faceted)",
    },
    "timecourse_pi": {
        "family": FAMILY_TIMECOURSE, "metric": "PI",
        "y_label": "Preference index (well A)",
        "y_limits": (-1.0, 1.0), "ref_line": 0.0,
        "display": "Preference index over time", "layout": "two_well",
    },
    "timecourse_licks": {
        "family": FAMILY_TIMECOURSE, "metric": "Licks", "y_label": "Licks per bin",
        "y_limits": None, "ref_line": None,
        "display": "Licks over time",
    },
    "timecourse_events": {
        "family": FAMILY_TIMECOURSE, "metric": "Events",
        "y_label": "Feeding events per bin",
        "y_limits": None, "ref_line": None,
        "display": "Events over time",
    },
    "timecourse_medduration": {
        "family": FAMILY_TIMECOURSE, "metric": "MedDuration",
        "y_label": "Median bout duration (s)",
        "y_limits": None, "ref_line": None,
        "display": "Median duration over time",
    },
    ## The Progressive Ratio headline: paired minus yoked cumulative sucrose-
    ## well licks against minutes since each Chamber Group's training end
    ## (ADR-0013).  Drawn from the Members' ``pr_cumulative_diff.csv`` rather
    ## than the binned summary, and offered only to that type.
    "timecourse_pr_diff": {
        "family": FAMILY_TIMECOURSE, "metric": "DiffCumLicks",
        "y_label": "Paired − yoked cumulative licks (well A)",
        "x_label": "Time since training end (min)",
        "y_limits": None, "ref_line": 0.0,
        "display": "Paired − yoked cumulative licks since training",
        "layout": "two_well", "module": "ratio",
        "source": "pr_diff", "common_range": True,
    },
    ## One point per Chamber Group: the paired fly's Breaking Point against
    ## the Test-phase minute of its last counted light event (ADR-0014).
    "scatter_pr_breaking_point": {
        "family": FAMILY_SCATTER, "metric": "BreakingPoint",
        "y_label": "Breaking point",
        "x_label": "Time since training end (min)",
        "y_limits": None, "ref_line": None,
        "display": "Breaking point vs Test time",
        "layout": "two_well", "module": "ratio",
        "source": "pr_breaking", "builder": "breaking_time",
    },
    ## One point per Chamber Group: Breaking Point by Treatment (no facets).
    "dot_pr_breaking_point": {
        "family": FAMILY_SCATTER, "metric": "BreakingPoint",
        "y_label": "Breaking point",
        "x_label": "",
        "y_limits": None, "ref_line": None,
        "display": "Breaking point by treatment",
        "layout": "two_well", "module": "ratio",
        "source": "pr_breaking", "builder": "breaking_treatment",
    },
}

#: The light table's numeric columns as faceted figures (decision 17 of
#: ``docs/v4-data-plan.md``): ``faceted_<metric lower-cased>``, drawn from the
#: Combined Analysis' stacked light table (:data:`SOURCE_LIGHT`), one point per
#: chamber, each labelled with its unit.  The four headline metrics of before
#: come first; then the other columns, and the Paradigm Families' own
#: (stream S2 of ``docs/paradigm-analyses-plan.md``).
LIGHT_METRICS: tuple[tuple[str, str], ...] = (
    ("Activations", "Activations"),
    ("LitSec", "Lit time (s)"),
    ("FirmwareEvents", "Firmware Events"),
    ("AcclimationEndMin", "Acclimation end (min)"),
    ("LitFraction", "Fraction of the time lit"),
    ("ActivationsPerHour", "Activations per hour"),
    ("LightOnsets", "Light onsets"),
    ("AcclimationActivations", "Acclimation Activations"),
    ("LastRequirementMet", "Last Requirement met"),
    ("ProgressAtEnd", "Progress at the end"),
    ("EventsPerActivation", "Firmware Events per Activation"),
    ("LockedOutContactSec", "Feeding while locked out (s)"),
    ("UnlitSec", "Unlit time (s)"),
    ("Suppressions", "Suppressions"),
    ("MainActivations", "Main Activations"),
    ("MedianActivationSec", "Median Activation (s)"),
    ("CutOffs", "Cut-offs"),
    ("MaxActivations", "Most Activations possible"),
    ("RewardEfficiency", "Reward efficiency (fraction)"),
    ("MedianReturnSec", "Median return after the lockout (s)"),
    ("MedianSuppressionSec", "Median suppression (s)"),
    ("LicksPerMinLit", "Licks in the light (per min)"),
    ("LicksPerMinDark", "Licks in the dark (per min)"),
    ("LightFeedingIndex", "Light feeding index"),
)
#: A light metric's fixed axis and reference line, where its range is known.
_LIGHT_AXES: dict[str, tuple[tuple[float, float], float]] = {
    "LightFeedingIndex": ((-1.0, 1.0), 0.0),
}
#: The named source of the light figures.
SOURCE_LIGHT = "light"
for _metric, _label in LIGHT_METRICS:
    _limits, _ref = _LIGHT_AXES.get(_metric, (None, None))
    PLOT_TYPES[f"faceted_{_metric.lower()}"] = {
        "family": FAMILY_FACETED, "metric": _metric, "y_label": _label,
        "y_limits": _limits, "ref_line": _ref, "free_y": _limits is None,
        "display": f"{_label} (light, faceted)", "source": SOURCE_LIGHT,
    }
del _metric, _label, _limits, _ref

#: Member file behind each named extra source, relative to ``analysis/``.
#: The default time-course source is ``binned_feeding_summary.csv``; the
#: default faceted source is the Combined Analysis facet table.
SOURCE_FILES: dict[str, str] = {
    "pr_diff": "pr_cumulative_diff.csv",
    "pr_breaking": "pr_breaking_point.csv",
}


def family_of(plot_id: str) -> str:
    return PLOT_TYPES.get(plot_id, {}).get("family", FAMILY_FACETED)


def _module_ids(modules) -> set[str]:
    """The Analysis Module ids *modules* names: an Experiment Type (its
    ``module_ids``), an iterable of ids, or ``None`` for none."""
    if modules is None:
        return set()
    ids = getattr(modules, "module_ids", modules)
    if isinstance(ids, str):
        return {ids.lower()}
    return {str(m).lower() for m in ids}


def plots_for_layout(chamber_layout: str, modules=None, *,
                     light: bool = False) -> list[str]:
    """Plot ids valid for *chamber_layout* — a PI figure is meaningless on a
    single-well recording, so the Editor never offers one.  A plot that names
    an Analysis Module (``module``) is offered only when *modules* (an
    Experiment Type, or its module ids) includes it (ADR-0019).  The light
    figures are offered only with *light* (a Project with a light table,
    :func:`has_light_table`)."""
    out = []
    for plot_id, info in PLOT_TYPES.items():
        if not plot_allowed(plot_id, chamber_layout, modules):
            continue
        if info.get("source") == SOURCE_LIGHT and not light:
            continue
        out.append(plot_id)
    return out


def plot_allowed(plot_id: str, chamber_layout: str | None, modules=None) -> bool:
    """Whether *plot_id* applies to this layout and the type's modules."""
    info = PLOT_TYPES.get(plot_id)
    if info is None:
        return False
    needed = info.get("layout")
    if needed is not None and chamber_layout is not None and needed != chamber_layout:
        return False
    needed_module = info.get("module")
    if needed_module is not None and needed_module not in _module_ids(modules):
        return False
    return True


def source_of(plot_id: str) -> str:
    """``"facet"``, ``"binned"`` or a named extra source (``"pr_diff"``)."""
    info = PLOT_TYPES.get(plot_id, {})
    if info.get("source"):
        return str(info["source"])
    return "binned" if info.get("family") == FAMILY_TIMECOURSE else "facet"


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

@dataclass
class PlotStyle:
    """A named, reusable look shared by every figure that references it.

    Field names match PyTrackingAnalysis's PlotStyle so a style block is
    portable between the two apps (see MIRRORED.md).
    """

    width_mm: float = 180.0
    height_mm: float = 70.0
    #: Per-facet panel width (mm). 0 = off (panels share ``width_mm``).
    facet_width_mm: float = 0.0
    #: Panel height (mm). 0 = off (the figure is ``height_mm`` tall).
    facet_height_mm: float = 0.0
    theme: str = "classic"            # classic | bw | minimal
    font_family: str = "Arial"
    base_pt: float = 8.0
    text_color: str = "#000000"
    point_size: float = 1.6
    point_alpha: float = 1.0
    jitter_width: float = 0.18
    mean_style: str = "point+sem"     # point+sem | bar+sem | point+95ci | bar+95ci
    mean_color: str = "#111111"
    point_stroke: float = 0.0
    geom: str = "dots"                # dots | box | box+dots
    strip_style: str = "plain"        # plain | boxed
    strip_bg: str = "#d9d9d9"
    panel_bg: str = ""
    line_pt: float = 0.8
    #: Time-course only: line weight and whether to draw the SEM ribbon.
    line_size: float = 0.7
    ribbon_alpha: float = 0.18
    #: treatment name -> hex color; unmapped treatments cycle the palette.
    colors: dict = field(default_factory=dict)
    palette: list = field(default_factory=lambda: list(DEFAULT_PALETTE))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlotStyle":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        style = cls(**{k: v for k, v in data.items() if k in known})
        if style.theme not in _THEMES:
            style.theme = "classic"
        if style.mean_style not in _MEAN_STYLES:
            style.mean_style = "point+sem"
        if style.geom not in _GEOMS:
            style.geom = "dots"
        if style.strip_style not in _STRIP_STYLES:
            style.strip_style = "plain"
        style.colors = dict(style.colors or {})
        style.palette = list(style.palette or DEFAULT_PALETTE)
        return style

    def color_for(self, treatment: str, index: int) -> str:
        explicit = (self.colors or {}).get(str(treatment))
        if explicit:
            return str(explicit)
        palette = self.palette or DEFAULT_PALETTE
        return palette[index % len(palette)]


@dataclass
class PlotSpec:
    """One figure's content decisions, plus the name of its style.

    Two shapes share the class: ``facets``/``facet_labels`` belong to a faceted
    metric plot, ``binsize``/``smooth`` to a time course.  Keeping one class
    means one yaml schema and one Inspector; the irrelevant fields are simply
    ignored by the other family's builder.
    """

    style: str = "default"
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    #: Faceted family: facet labels to include, in order; empty = all.
    facets: list | None = None
    facet_labels: dict = field(default_factory=dict)
    #: Faceted family with a Role column (Progressive Ratio): roles to include
    #: as top columns (``paired`` / ``yoked``), in order; empty = all.  Ignored
    #: when the tidy frame has no Role — non-PR figures stay phase-only.
    roles: list | None = None
    #: treatment name -> {"label": display, "show": bool}; dict order = plot order.
    treatments: dict = field(default_factory=dict)
    y_limits: list | None = None
    ref_line: float | None = None
    free_y: bool = False
    #: Give each Member its own point shape so batch structure is visible
    #: in the pooled figure.  Ignored when the data has no Experiment column.
    mark_experiments: bool = False
    #: Time-course family: bin width in minutes, and whether to draw the SEM
    #: ribbon around each treatment's mean.
    binsize: float = 30.0
    ribbon: bool = True
    #: Time-course family: draw the mean only over the x range every series
    #: covers, so a series ending early never changes the average by leaving
    #: it (cumulative curves since a per-group training end).
    common_range: bool = False
    #: The Slice whose Combined Analysis the figure draws from (ADR-0020), or
    #: ``None`` for the whole recording.  Stated in the Spec so the figure is
    #: regenerable from it; rendered into ``slices/<name>/figures/``.
    slice: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "PlotSpec":
        data = dict(data or {})
        known = set(cls.__dataclass_fields__)
        spec = cls(**{k: v for k, v in data.items() if k in known})
        spec.facet_labels = dict(spec.facet_labels or {})
        spec.treatments = {
            str(name): {"label": str((entry or {}).get("label", name)),
                        "show": bool((entry or {}).get("show", True))}
            for name, entry in (spec.treatments or {}).items()
        }
        if spec.y_limits is not None:
            spec.y_limits = [float(v) for v in spec.y_limits]
        spec.slice = str(spec.slice).strip() or None if spec.slice is not None else None
        return spec


def default_spec(plot_id: str, well_a: str = "well A") -> PlotSpec:
    info = PLOT_TYPES[plot_id]
    y_limits = info["y_limits"]
    if info["family"] == FAMILY_FACETED or info.get("builder") == "breaking_treatment":
        x_label = str(info.get("x_label") or "")
    else:
        x_label = str(info.get("x_label") or "Time (min)")
    return PlotSpec(
        y_label=str(info["y_label"]).replace("well A", well_a),
        y_limits=list(y_limits) if y_limits is not None else None,
        ref_line=info["ref_line"],
        free_y=bool(info.get("free_y", False)),
        x_label=x_label,
        common_range=bool(info.get("common_range", False)),
    )


# --------------------------------------------------------------------------
# plot_specs.yaml
# --------------------------------------------------------------------------

@dataclass
class ProjectSpecs:
    """The parsed ``plot_specs.yaml``."""

    default_style: str = "default"
    styles: dict = field(default_factory=dict)   # name -> PlotStyle
    plots: dict = field(default_factory=dict)    # plot_id -> PlotSpec

    def style_for(self, spec: PlotSpec) -> PlotStyle:
        return (self.styles.get(spec.style)
                or self.styles.get(self.default_style)
                or PlotStyle())

    def ensure_default_style(self) -> None:
        if not self.styles:
            self.styles["default"] = PlotStyle()
        if self.default_style not in self.styles:
            self.default_style = next(iter(self.styles))


def specs_path(project_dir: str) -> str:
    return os.path.join(str(project_dir), SPECS_FILENAME)


def load_project_specs(project_dir: str) -> ProjectSpecs:
    path = specs_path(project_dir)
    raw: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    specs = ProjectSpecs(
        default_style=str(raw.get("default_style", "default")),
        styles={str(k): PlotStyle.from_dict(v)
                for k, v in (raw.get("styles") or {}).items()},
        plots={str(k): PlotSpec.from_dict(v)
               for k, v in (raw.get("plots") or {}).items()
               if str(k) in PLOT_TYPES},
    )
    specs.ensure_default_style()
    return specs


def save_project_specs(project_dir: str, specs: ProjectSpecs) -> str:
    specs.ensure_default_style()
    payload = {
        "default_style": specs.default_style,
        "styles": {k: v.to_dict() for k, v in specs.styles.items()},
        "plots": {k: v.to_dict() for k, v in specs.plots.items()},
    }
    path = specs_path(project_dir)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return path


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _metric_values(df: pd.DataFrame, metric: str, *, transformed: bool = False) -> pd.Series:
    from .analytics import _resolve_metric_col

    return pd.to_numeric(_resolve_metric_col(df, metric, transformed=transformed),
                         errors="coerce")


def licks_transformed(project, plot_id: str) -> bool:
    """Whether *plot_id*'s source table holds Licks on the fourth-root
    scale: the feeding summaries (Facet and binned) follow the Design's
    ``transform_licks``, so a two-well Licks total is (A + B)^¼."""
    return bool(project.transform_licks) and source_of(plot_id) in ("facet", "binned")


def _panels_roles(frame: pd.DataFrame) -> bool:
    """Whether *frame*'s ``Role`` is Progressive Ratio's paired / yoked (the
    one Role a figure panels by): every Role it names is one of
    :data:`ROLE_ORDER`."""
    if "Role" not in frame.columns:
        return False
    roles = frame["Role"].fillna("").astype(str).str.strip()
    named = set(roles[(roles != "") & (roles.str.lower() != "nan")])
    return bool(named) and named <= set(ROLE_ORDER)


def faceted_data(frame: pd.DataFrame, metric: str,
                 label_order: list[str] | None = None, *,
                 transformed: bool = False) -> pd.DataFrame:
    """Tidy per-chamber data for one metric: Treatment, Phase, Value
    [, Role, Experiment].

    *frame* is a faceted summary (a Member's ``feeding_summary_facet.csv`` or
    a Project's ``_Summary_Facet.csv``).  An unfaceted frame becomes a single
    "Whole recording" phase, so the same builder serves both.  A Progressive
    Ratio frame's ``Role`` column is kept so the figure can panel paired and
    yoked separately instead of pooling them.  Any other ``Role`` (the
    optogenetic chamber Role every optogenetic summary carries: Paired,
    Self, Open, …) is a column to filter on, not a panel, and is dropped.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Treatment", "Phase", "Value"])
    if "Facet" in frame.columns:
        phase = frame["Facet"].astype(str)
        order = label_order or list(dict.fromkeys(phase))
    else:
        from .slices import WHOLE_RECORDING_LABEL

        phase = pd.Series([WHOLE_RECORDING_LABEL] * len(frame), index=frame.index)
        order = [WHOLE_RECORDING_LABEL]
    out = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "Phase": pd.Categorical(phase, categories=order, ordered=True),
        "Value": _metric_values(frame, metric, transformed=transformed),
    })
    if _panels_roles(frame):
        roles = frame["Role"].astype(str).str.strip()
        role_order = order_roles(roles)
        out["Role"] = pd.Categorical(roles, categories=role_order, ordered=True)
    if "Experiment" in frame.columns:
        out["Experiment"] = frame["Experiment"].astype(str)
    out = out[(out["Treatment"] != "") & out["Value"].notna()]
    if "Role" in out.columns:
        out = out[out["Role"].notna()
                  & (out["Role"].astype(str).str.strip() != "")
                  & (out["Role"].astype(str).str.lower() != "nan")]
    return out.reset_index(drop=True)


def timecourse_data(frame: pd.DataFrame, metric: str, *,
                    transformed: bool = False) -> pd.DataFrame:
    """Tidy per-chamber binned data: Treatment, Minutes, Value [, Experiment].

    *frame* is a binned summary (``binned_feeding_summary.csv``), which carries
    one row per chamber per time bin with a ``Minutes`` bin-midpoint column.
    """
    if frame is None or frame.empty or "Minutes" not in frame.columns:
        return pd.DataFrame(columns=["Treatment", "Minutes", "Value"])
    out = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "Minutes": pd.to_numeric(frame["Minutes"], errors="coerce"),
        "Value": _metric_values(frame, metric, transformed=transformed),
    })
    if "Experiment" in frame.columns:
        out["Experiment"] = frame["Experiment"].astype(str)
    out = out[(out["Treatment"] != "") & out["Value"].notna()
              & out["Minutes"].notna()]
    return out.reset_index(drop=True)


def data_treatments(df: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for value in df["Treatment"]:
        if value not in seen:
            seen.append(value)
    return seen


def merged_treatments(spec: PlotSpec, df: pd.DataFrame) -> dict:
    """The spec's treatment table, extended with any treatment present in the
    data but not yet named — so a new treatment appears rather than vanishing."""
    merged = {name: dict(entry) for name, entry in (spec.treatments or {}).items()}
    for name in data_treatments(df):
        merged.setdefault(name, {"label": name, "show": True})
    return merged


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_AXIS_MARGIN_MM = 22.0
_PT_TO_MM = 25.4 / 72.0


#: Family names matplotlib resolves itself, never looked up as installed faces.
_GENERIC_FAMILIES = {"sans-serif", "serif", "monospace", "cursive", "fantasy"}


@lru_cache(maxsize=1)
def _installed_fonts() -> frozenset[str]:
    from matplotlib import font_manager

    return frozenset(f.name for f in font_manager.fontManager.ttflist)


def resolve_font_family(requested: str) -> list[str]:
    """*requested* first, with generic fallbacks, so a missing face degrades to
    something sane instead of matplotlib's default at some sizes only.

    Only faces this machine has are listed (and the generic families).
    matplotlib logs "Font family … not found" for every listed face it lacks,
    on every text it draws, so a fallback missing here (Liberation Sans on
    Windows) flooded the console with warnings during a Project Report."""
    installed = _installed_fonts()

    def usable(name: str) -> bool:
        return name in installed or name.lower() in _GENERIC_FAMILIES

    wanted = str(requested or "").strip()
    fallbacks = [f for f in ("DejaVu Sans", "Liberation Sans") if usable(f)] + ["sans-serif"]
    if wanted and usable(wanted):
        return [wanted] + [f for f in fallbacks if f != wanted]
    return fallbacks


def effective_width_mm(style: PlotStyle, n_facets: int | None = None) -> float:
    if style.facet_width_mm and style.facet_width_mm > 0 and n_facets:
        return _AXIS_MARGIN_MM + float(style.facet_width_mm) * int(n_facets)
    return float(style.width_mm)


def effective_height_mm(style: PlotStyle, spec: PlotSpec | None = None,
                        n_rows: int | None = None) -> float:
    if not (style.facet_height_mm and style.facet_height_mm > 0):
        return float(style.height_mm)
    line_mm = float(style.base_pt) * _PT_TO_MM
    margin = 10.0 + 2.5 * line_mm
    if spec is not None:
        if (spec.x_label or "").strip():
            margin += 1.8 * line_mm
        if (spec.title or "").strip():
            margin += 1.8 * (line_mm + 2 * _PT_TO_MM)
    rows = max(int(n_rows or 1), 1)
    return float(style.facet_height_mm) * rows + margin


def _theme_for(style: PlotStyle, n_facets: int | None = None,
               height_mm: float | None = None, show_legend: bool = False):
    import plotnine as p9

    base = {"classic": p9.theme_classic, "bw": p9.theme_bw,
            "minimal": p9.theme_minimal}.get(style.theme, p9.theme_classic)
    families = resolve_font_family(style.font_family)
    lw = float(style.line_pt)
    ink = style.text_color or "#000000"
    overrides = dict(
        text=p9.element_text(family=families, color=ink),
        figure_size=(effective_width_mm(style, n_facets) / 25.4,
                     (height_mm if height_mm is not None
                      else style.height_mm) / 25.4),
        legend_position="bottom" if show_legend else "none",
        legend_title=p9.element_text(size=style.base_pt, color=ink),
        strip_text=p9.element_text(size=style.base_pt + 1, color=ink),
        plot_title=p9.element_text(size=style.base_pt + 2, color=ink),
        axis_text=p9.element_text(color=ink),
        axis_line=p9.element_line(size=lw),
        axis_ticks=p9.element_line(size=lw),
    )
    if style.strip_style == "boxed":
        overrides["strip_background"] = p9.element_rect(
            fill=style.strip_bg or "#d9d9d9", color="black", size=lw)
        overrides["panel_border"] = p9.element_rect(color="black", size=lw,
                                                    fill=None)
    else:
        overrides["strip_background"] = p9.element_blank()
    if style.panel_bg:
        overrides["panel_background"] = p9.element_rect(fill=style.panel_bg)
    ## base_family pins the theme's own defaults to the SAME resolved face the
    ## text themeable uses; without it some elements fall back to DejaVu while
    ## others honour the request, which reads as a condensed font at some sizes.
    return base(base_size=style.base_pt,
                base_family=families[0]) + p9.theme(**overrides)


def _apply_treatment_order(data: pd.DataFrame, spec: PlotSpec, style: PlotStyle):
    """Returns ``(data, labels, colors)`` with Treatment recoded to display
    labels in spec order.  Colors stay keyed by the *original* name, so
    renaming a treatment never changes its color."""
    treatments = merged_treatments(spec, data)
    order = [n for n, e in treatments.items() if e.get("show", True)]
    data = data[data["Treatment"].isin(order)].copy()
    labels = [str(treatments[n].get("label", n)) for n in order]
    colors = [style.color_for(n, i) for i, n in enumerate(order)]
    data["Treatment"] = pd.Categorical(
        data["Treatment"].astype(str).map(
            {n: str(treatments[n].get("label", n)) for n in order}),
        categories=labels, ordered=True)
    return data, labels, colors


def build_faceted(df: pd.DataFrame, spec: PlotSpec, style: PlotStyle):
    """Per-treatment jittered points with a mean overlay, one panel per Facet.

    When the tidy frame carries ``Role`` (Progressive Ratio), panels are a
    ``Phase ~ Role`` grid so paired and yoked are never pooled: Roles along
    the top, phases down the right in the order the experiment ran them
    (Training above Test).  ``spec.roles`` and ``spec.facets``
    independently choose which columns and rows appear.
    """
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
    all_phases = list(data["Phase"].cat.categories)
    include = [p for p in (spec.facets or all_phases) if p in all_phases]
    data = data[data["Phase"].isin(include)].copy()
    shown = [str(spec.facet_labels.get(p, p)) for p in include]
    data["Phase"] = pd.Categorical(
        data["Phase"].astype(str).map(lambda p: str(spec.facet_labels.get(p, p))),
        categories=shown, ordered=True)

    by_role = "Role" in data.columns and not data.empty
    shown_roles: list[str] = []
    if by_role:
        if hasattr(data["Role"], "cat"):
            all_roles = [str(r) for r in data["Role"].cat.categories]
        else:
            all_roles = list(dict.fromkeys(data["Role"].astype(str)))
        selected = set(spec.roles) if spec.roles else set(all_roles)
        shown_roles = order_roles(r for r in all_roles if r in selected)
        data = data[data["Role"].astype(str).isin(shown_roles)].copy()
        ## Canonical roles stay lowercase in the Spec / CSV; strips print
        ## Yoked / Paired.
        role_labels = [role_display_label(r) for r in shown_roles]
        data["Role"] = pd.Categorical(
            data["Role"].astype(str).map(role_display_label),
            categories=role_labels, ordered=True)
        by_role = bool(shown_roles) and not data.empty

    mark = bool(spec.mark_experiments) and "Experiment" in data.columns
    scales = "free_y" if spec.free_y else "fixed"
    if by_role:
        facet = p9.facet_grid("Phase ~ Role", scales=scales)
    else:
        facet = p9.facet_wrap("~Phase", nrow=1, scales=scales)
    g = (p9.ggplot(data, p9.aes("Treatment", "Value", color="Treatment"))
         + facet
         + p9.scale_color_manual(values=colors)
         + p9.labs(title=spec.title or "", x=spec.x_label or "",
                   y=spec.y_label or ""))
    if spec.ref_line is not None:
        g = g + p9.geom_hline(yintercept=float(spec.ref_line),
                              linetype="dashed", color="#888888", size=0.3)

    boxed = style.geom in ("box", "box+dots")
    if boxed:
        g = g + p9.geom_boxplot(
            fill="white", width=0.6, size=style.line_pt * 0.9,
            outlier_size=(0 if style.geom == "box+dots"
                          else max(style.point_size * 0.8, 0.5)),
            outlier_alpha=style.point_alpha)
    if style.geom in ("dots", "box+dots"):
        stroke = max(0.0, float(style.point_stroke))
        point_aes = {"shape": "Experiment"} if mark else {}
        if stroke > 0:
            g = (g + p9.geom_jitter(p9.aes(fill="Treatment", **point_aes),
                                    color="black", stroke=stroke,
                                    width=style.jitter_width, height=0,
                                    size=style.point_size,
                                    alpha=style.point_alpha, random_state=0)
                 + p9.scale_fill_manual(values=colors))
        else:
            g = g + p9.geom_jitter(p9.aes(**point_aes) if point_aes else None,
                                   width=style.jitter_width, height=0,
                                   size=style.point_size,
                                   alpha=style.point_alpha, random_state=0)
    if mark:
        ## Only the shape (member) legend is useful — treatments are named
        ## on the x axis already.
        g = g + p9.guides(color="none", fill="none")

    if not boxed:
        group_cols = (["Phase", "Role", "Treatment"] if by_role
                      else ["Phase", "Treatment"])
        stat = data.groupby(group_cols, observed=True)["Value"].agg(
            ["mean", "sem"]).reset_index()
        mult = 1.96 if "95ci" in style.mean_style else 1.0
        stat["ymin"] = stat["mean"] - mult * stat["sem"].fillna(0.0)
        stat["ymax"] = stat["mean"] + mult * stat["sem"].fillna(0.0)
        if style.mean_style.startswith("bar"):
            g = g + p9.geom_col(
                p9.aes(x="Treatment", y="mean"), data=stat, inherit_aes=False,
                fill="none", color=style.mean_color, width=0.55,
                size=style.line_pt)
        else:
            g = g + p9.geom_point(
                p9.aes(x="Treatment", y="mean"), data=stat, inherit_aes=False,
                color=style.mean_color, size=style.point_size * 1.9,
                shape="_")
        g = g + p9.geom_errorbar(
            p9.aes(x="Treatment", ymin="ymin", ymax="ymax"), data=stat,
            inherit_aes=False, color=style.mean_color, width=0.22,
            size=style.line_pt)

    if spec.y_limits and not spec.free_y:
        g = g + p9.coord_cartesian(ylim=tuple(float(v) for v in spec.y_limits))
    ## Width follows columns (Roles when present, else phases); height follows
    ## rows (phases when Role is present).
    n_cols = max(len(shown_roles), 1) if by_role else max(len(shown), 1)
    n_rows = max(len(shown), 1) if by_role else 1
    return g + _theme_for(style, n_facets=n_cols,
                          height_mm=effective_height_mm(style, spec,
                                                        n_rows=n_rows),
                          show_legend=mark)


def common_range_rows(data: pd.DataFrame) -> pd.DataFrame:
    """The rows of a tidy time course at bins every series contributes to.

    A tidy frame carries no series id, but each series has exactly one row per
    bin, so the bins with the most rows are the ones every series reaches;
    keeping only those truncates the average to the shortest series instead
    of letting it jump when one series ends.
    """
    if data is None or data.empty or "Minutes" not in data.columns:
        return data
    counts = data.groupby("Minutes").size()
    full = counts[counts == counts.max()].index
    return data[data["Minutes"].isin(full)]


def build_timecourse(df: pd.DataFrame, spec: PlotSpec, style: PlotStyle):
    """One line per treatment over time bins, with an SEM ribbon."""
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
    if getattr(spec, "common_range", False):
        data = common_range_rows(data)
    stat = data.groupby(["Minutes", "Treatment"], observed=True)["Value"].agg(
        ["mean", "sem"]).reset_index()
    stat["sem"] = stat["sem"].fillna(0.0)
    stat["ymin"] = stat["mean"] - stat["sem"]
    stat["ymax"] = stat["mean"] + stat["sem"]

    g = (p9.ggplot(stat, p9.aes("Minutes", "mean", color="Treatment"))
         + p9.scale_color_manual(values=colors)
         + p9.scale_fill_manual(values=colors)
         + p9.labs(title=spec.title or "", x=spec.x_label or "Time (min)",
                   y=spec.y_label or ""))
    if spec.ref_line is not None:
        g = g + p9.geom_hline(yintercept=float(spec.ref_line),
                              linetype="dashed", color="#888888", size=0.3)
    if spec.ribbon:
        g = g + p9.geom_ribbon(
            p9.aes(ymin="ymin", ymax="ymax", fill="Treatment"),
            alpha=float(style.ribbon_alpha), color="none")
    g = g + p9.geom_line(size=float(style.line_size))
    if style.geom in ("dots", "box+dots"):
        g = g + p9.geom_point(size=style.point_size, alpha=style.point_alpha)
    if spec.y_limits:
        g = g + p9.coord_cartesian(ylim=tuple(float(v) for v in spec.y_limits))
    ## A time course has no x-axis treatment names, so its legend is the only
    ## way to read the lines — always shown.
    return g + _theme_for(style, n_facets=1,
                          height_mm=effective_height_mm(style, spec),
                          show_legend=True)


def breaking_point_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Tidy Breaking Point rows: Treatment, BreakingPoint, Censored
    [, BreakMin, Experiment].

    *frame* is a stacked ``pr_breaking_point.csv`` (one row per chamber group).
    ``BreakMin`` is kept when present (time scatter); treatment plots need only
    Treatment and BreakingPoint.
    """
    cols = ["Treatment", "BreakMin", "BreakingPoint", "Censored"]
    if frame is None or frame.empty:
        return pd.DataFrame(columns=cols)
    need = {"Treatment", "BreakingPoint"}
    if not need.issubset(frame.columns):
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "BreakingPoint": pd.to_numeric(frame["BreakingPoint"], errors="coerce"),
    })
    if "BreakMin" in frame.columns:
        out["BreakMin"] = pd.to_numeric(frame["BreakMin"], errors="coerce")
    else:
        out["BreakMin"] = pd.NA
    if "Censored" in frame.columns:
        from .analytics import as_bool
        out["Censored"] = as_bool(frame["Censored"])
    else:
        out["Censored"] = False
    if "Experiment" in frame.columns:
        out["Experiment"] = frame["Experiment"].astype(str)
    out = out[(out["Treatment"] != "") & out["BreakingPoint"].notna()]
    return out.reset_index(drop=True)


def build_breaking_point_scatter(df: pd.DataFrame, spec: PlotSpec,
                                 style: PlotStyle):
    """One point per chamber group: Breaking Point vs Test-phase minutes.

    Open symbols are censored (still responding at the end of the Test window);
    filled symbols are observed breaks.
    """
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
    if "BreakMin" in data.columns:
        data = data[data["BreakMin"].notna()].copy()
    if data.empty:
        return (p9.ggplot()
                + p9.labs(title=spec.title or "Breaking point vs Test time",
                          x=spec.x_label or "", y=spec.y_label or "")
                + _theme_for(style, n_facets=1,
                             height_mm=effective_height_mm(style, spec),
                             show_legend=False))

    if "Censored" in data.columns:
        censored = data["Censored"].fillna(False).astype(bool)
    else:
        censored = pd.Series(False, index=data.index)
    mark = bool(spec.mark_experiments) and "Experiment" in data.columns
    point_aes = {"shape": "Experiment"} if mark else {}
    observed = data.loc[~censored]
    open_pts = data.loc[censored]
    g = (p9.ggplot(data, p9.aes("BreakMin", "BreakingPoint", color="Treatment"))
         + p9.scale_color_manual(values=colors)
         + p9.labs(title=spec.title or "",
                   x=spec.x_label or "Time since training end (min)",
                   y=spec.y_label or "Breaking point",
                   caption=("Open: censored (still responding at Test end)"
                            if bool(censored.any()) else None)))
    if not observed.empty:
        g = g + p9.geom_point(
            p9.aes(**point_aes) if point_aes else None, data=observed,
            size=style.point_size * 1.8, alpha=style.point_alpha)
    if not open_pts.empty:
        stroke = max(float(style.point_stroke), 0.9)
        g = g + p9.geom_point(
            p9.aes(**point_aes) if point_aes else None, data=open_pts,
            fill="white", stroke=stroke,
            size=style.point_size * 1.8, alpha=style.point_alpha)
    if spec.ref_line is not None:
        g = g + p9.geom_hline(yintercept=float(spec.ref_line),
                              linetype="dashed", color="#888888", size=0.3)
    if spec.y_limits:
        g = g + p9.coord_cartesian(ylim=tuple(float(v) for v in spec.y_limits))
    ## Treatment is not on the x axis — the colour legend is required.
    return g + _theme_for(style, n_facets=1,
                          height_mm=effective_height_mm(style, spec),
                          show_legend=True)


def build_breaking_point_by_treatment(df: pd.DataFrame, spec: PlotSpec,
                                      style: PlotStyle):
    """One point per chamber group: Breaking Point by Treatment (no facets).

    Open symbols are censored; filled symbols are observed.  Mean ± SEM overlays
    each treatment when the style's mean form is not a box plot.
    """
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
    if data.empty:
        return (p9.ggplot()
                + p9.labs(title=spec.title or "Breaking point by treatment",
                          x=spec.x_label or "", y=spec.y_label or "")
                + _theme_for(style, n_facets=1,
                             height_mm=effective_height_mm(style, spec),
                             show_legend=False))

    if "Censored" in data.columns:
        censored = data["Censored"].fillna(False).astype(bool)
    else:
        censored = pd.Series(False, index=data.index)
    mark = bool(spec.mark_experiments) and "Experiment" in data.columns
    point_aes = {"shape": "Experiment"} if mark else {}
    observed = data.loc[~censored]
    open_pts = data.loc[censored]
    g = (p9.ggplot(data, p9.aes("Treatment", "BreakingPoint", color="Treatment"))
         + p9.scale_color_manual(values=colors)
         + p9.labs(title=spec.title or "",
                   x=spec.x_label or "",
                   y=spec.y_label or "Breaking point",
                   caption=("Open: censored (still responding at Test end)"
                            if bool(censored.any()) else None)))
    boxed = style.geom in ("box", "box+dots")
    if boxed:
        g = g + p9.geom_boxplot(
            fill="white", width=0.6, size=style.line_pt * 0.9,
            outlier_size=0, outlier_alpha=0)
    if style.geom in ("dots", "box+dots") or not boxed:
        if not observed.empty:
            g = g + p9.geom_jitter(
                p9.aes(**point_aes) if point_aes else None, data=observed,
                width=style.jitter_width, height=0,
                size=style.point_size * 1.8, alpha=style.point_alpha,
                random_state=0)
        if not open_pts.empty:
            stroke = max(float(style.point_stroke), 0.9)
            g = g + p9.geom_jitter(
                p9.aes(**point_aes) if point_aes else None, data=open_pts,
                width=style.jitter_width, height=0,
                fill="white", stroke=stroke,
                size=style.point_size * 1.8, alpha=style.point_alpha,
                random_state=1)
    if mark:
        g = g + p9.guides(color="none")
    if not boxed:
        stat = data.groupby(["Treatment"], observed=True)["BreakingPoint"].agg(
            ["mean", "sem"]).reset_index()
        mult = 1.96 if "95ci" in style.mean_style else 1.0
        stat["ymin"] = stat["mean"] - mult * stat["sem"].fillna(0.0)
        stat["ymax"] = stat["mean"] + mult * stat["sem"].fillna(0.0)
        if style.mean_style.startswith("bar"):
            g = g + p9.geom_col(
                p9.aes(x="Treatment", y="mean"), data=stat, inherit_aes=False,
                fill="none", color=style.mean_color, width=0.55,
                size=style.line_pt)
        else:
            g = g + p9.geom_point(
                p9.aes(x="Treatment", y="mean"), data=stat, inherit_aes=False,
                color=style.mean_color, size=style.point_size * 1.9,
                shape="_")
        g = g + p9.geom_errorbar(
            p9.aes(x="Treatment", ymin="ymin", ymax="ymax"), data=stat,
            inherit_aes=False, color=style.mean_color, width=0.22,
            size=style.line_pt)
    if spec.ref_line is not None:
        g = g + p9.geom_hline(yintercept=float(spec.ref_line),
                              linetype="dashed", color="#888888", size=0.3)
    if spec.y_limits:
        g = g + p9.coord_cartesian(ylim=tuple(float(v) for v in spec.y_limits))
    ## Treatments are named on the x axis — hide the colour legend unless
    ## member shapes need room beside it.
    return g + _theme_for(style, n_facets=1,
                          height_mm=effective_height_mm(style, spec),
                          show_legend=mark)


def build_figure(plot_id: str, df: pd.DataFrame, spec: PlotSpec,
                 style: PlotStyle):
    """Dispatch to the builder for *plot_id*'s family."""
    family = family_of(plot_id)
    if family == FAMILY_TIMECOURSE:
        return build_timecourse(df, spec, style)
    if family == FAMILY_SCATTER:
        builder = PLOT_TYPES.get(plot_id, {}).get("builder", "breaking_time")
        if builder == "breaking_treatment":
            return build_breaking_point_by_treatment(df, spec, style)
        return build_breaking_point_scatter(df, spec, style)
    return build_faceted(df, spec, style)


def render_png_bytes(g, style: PlotStyle, dpi: int = 120) -> bytes:
    """A PNG of *g* for on-screen preview.  Bytes rather than a file so the
    Editor never writes to disk to show a preview."""
    buffer = io.BytesIO()
    g.save(buffer, format="png", dpi=dpi, verbose=False)
    return buffer.getvalue()


def _vector_rc(fmt: str) -> dict:
    """Matplotlib settings that keep text editable in the saved vector file."""
    if fmt == "svg":
        return {"svg.fonttype": "none"}
    return {"pdf.fonttype": 42, "ps.fonttype": 42}


def save_figure(g, path: str, fmt: str = "svg") -> str:
    """Write *g* to *path* as an editable-text vector file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with matplotlib.rc_context(_vector_rc(fmt)):
        g.save(path, format=fmt, verbose=False)
    return path


# --------------------------------------------------------------------------
# Project-level rendering
# --------------------------------------------------------------------------

def _member_out(project, name: str) -> str:
    """A Member's outputs in the Project's view: its directory, or
    ``slices/<name>/`` inside it for a Slice (ADR-0020)."""
    return project.member_output_dir(name)


def _pooled(project) -> list[str]:
    """The Members the view pools: a Slice leaves out those that do not
    cover it."""
    return project._pooled_members()


def _read(project, path) -> pd.DataFrame:
    """A saved table with its Treatment and the design's factors read as
    text (:func:`pyflic.base.utils.read_result_csv`): a level named "None"
    or "NA" stays a level in the figures."""
    from .utils import read_result_csv

    factors = tuple(getattr(project, "design_factors", None) or ())
    return read_result_csv(path, ("Treatment", *factors))


def project_frames(project):
    """``(facet_frame, binned_frame)`` for a Project's pooled figures.

    The facet frame comes from the Combined Analysis; the binned frame is
    stacked here from each Member's saved ``binned_feeding_summary.csv``,
    because a time course needs bins the faceted summary does not carry.
    """
    facet_path = os.path.join(project.analysis_path,
                              f"{project.name}_Summary_Facet.csv")
    facet = _read(project, facet_path) if os.path.isfile(facet_path) else None
    if facet is None:
        summary_path = os.path.join(project.analysis_path,
                                    f"{project.name}_Summary.csv")
        facet = (_read(project, summary_path)
                 if os.path.isfile(summary_path) else None)

    binned_frames = []
    for name in _pooled(project):
        path = os.path.join(_member_out(project, name), "analysis",
                            "binned_feeding_summary.csv")
        if os.path.isfile(path):
            df = _read(project, path)
            df.insert(0, "Experiment", name)
            binned_frames.append(df)
    binned = pd.concat(binned_frames, ignore_index=True) if binned_frames else None
    return facet, binned


def project_source_frame(project, source: str):
    """A stacked per-Member frame for a named extra source (see
    :data:`SOURCE_FILES`), with an ``Experiment`` column, or ``None``."""
    filename = SOURCE_FILES.get(source)
    if filename is None:
        return None
    frames = []
    for name in _pooled(project):
        path = os.path.join(_member_out(project, name), "analysis", filename)
        if os.path.isfile(path):
            df = _read(project, path)
            df.insert(0, "Experiment", name)
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def facet_order(frame) -> list[str] | None:
    """*frame*'s Facet labels in first-seen order, or ``None`` unfaceted."""
    if frame is None or "Facet" not in frame.columns:
        return None
    return list(dict.fromkeys(frame["Facet"].astype(str)))


def _light_table_paths(project) -> list[str]:
    """The Combined Analysis' stacked light table, faceted first."""
    return [os.path.join(project.analysis_path, f"{project.name}_{stem}.csv")
            for stem in ("LightSummary_Facet", "LightSummary")]


def has_light_table(project) -> bool:
    """Whether the Project's Combined Analysis has a light table (one of its
    Members is optogenetic), so the light figures have something to draw."""
    return project is not None and any(os.path.isfile(p) for p in _light_table_paths(project))


def project_light_frame(project):
    """The light figures' source: the Combined Analysis' stacked per-chamber
    light table (``<project>_LightSummary_Facet.csv``, else the unfaceted
    ``<project>_LightSummary.csv``), or ``None``."""
    for path in _light_table_paths(project):
        if os.path.isfile(path):
            return _read(project, path)
    return None


def frame_for(plot_id: str, facet, binned, project=None):
    """The source frame a plot draws from, given the two standard frames and,
    for an extra source, the Project to stack it from."""
    source = source_of(plot_id)
    if source == "facet":
        return facet
    if source == "binned":
        return binned
    if source == SOURCE_LIGHT:
        return project_light_frame(project) if project is not None else None
    return project_source_frame(project, source) if project is not None else None


def render_all(project, fmt: str = "svg", out_dir: str | None = None,
               only: list[str] | None = None, log=print) -> list[str]:
    """Render every figure this Project can draw.

    Not "every Spec in ``plot_specs.yaml``": the Plot Editor edits one plot
    at a time and curates no subset, so a Spec records *how* a figure is
    drawn, never *whether* it is.  A plot the file does not mention is drawn
    from its default Spec, exactly as the Project Report already does.

    A plot whose data is unavailable (a time course with no saved binned
    summaries, a PI figure on a single-well Project) is skipped with a log
    line rather than failing the run — one missing figure must not abort an
    unattended Batch Run.
    """
    specs = load_project_specs(project.project_directory)
    whole = project.whole
    ## Each Spec names the Slice it draws from (ADR-0020); the frames of each
    ## view are read once.
    views: dict[str | None, tuple] = {}

    def view_of(name: str | None):
        if name not in views:
            view = whole if name is None else whole.for_slice(name)
            facet, binned = project_frames(view)
            ## The Facet column already carries each row's display label, so
            ## its first-seen order is the phase order — for fixed cutoffs and
            ## for a type whose windows come from the data alike (ADR-0013).
            label_order = None
            if facet is not None and "Facet" in facet.columns:
                label_order = list(dict.fromkeys(facet["Facet"].astype(str)))
            elif facet is not None and "FacetRange" in facet.columns:
                _, label_of = view.window_labels(facet)
                label_order = list(dict.fromkeys(label_of.values()))
            target = out_dir or (os.path.join(project.project_directory, FIGURES_DIRNAME)
                                 if name is None else view.figures_path)
            views[name] = (view, facet, binned, label_order, target)
        return views[name]

    well_a = str((project.design_global.get("well_names") or {}).get("A")
                 or "well A")
    written: list[str] = []
    ## plots_for_layout has already applied the layout and Analysis Module
    ## gates, so anything it lists is a figure this Project is entitled to.
    for plot_id in plots_for_layout(project.chamber_layout,
                                    project.experiment_type,
                                    light=has_light_table(project)):
        if only and plot_id not in only:
            continue
        spec = specs.plots.get(plot_id) or default_spec(plot_id, well_a)
        info = PLOT_TYPES[plot_id]
        metric = info["metric"]
        family = info["family"]
        try:
            view, facet, binned, label_order, target = view_of(spec.slice)
        except ValueError as err:
            log(f"    skipped {plot_id}: {err}")
            continue
        source = frame_for(plot_id, facet, binned, view)
        if source is None or source.empty:
            where = "" if spec.slice is None else f" in Slice {spec.slice}"
            log(f"    skipped {plot_id}: no {source_of(plot_id)} data{where}")
            continue
        transformed = licks_transformed(whole, plot_id)
        if family == FAMILY_TIMECOURSE:
            df = timecourse_data(source, metric, transformed=transformed)
        elif family == FAMILY_SCATTER:
            df = breaking_point_data(source)
        else:
            df = faceted_data(source, metric, label_order if source_of(plot_id) == "facet"
                              else facet_order(source), transformed=transformed)
        if df.empty:
            log(f"    skipped {plot_id}: metric '{metric}' not in the data")
            continue
        try:
            g = build_figure(plot_id, df, spec, specs.style_for(spec))
            os.makedirs(target, exist_ok=True)
            path = save_figure(g, os.path.join(target, f"{plot_id}.{fmt}"), fmt)
        except Exception as err:  # noqa: BLE001
            log(f"    skipped {plot_id}: {type(err).__name__}: {err}")
            continue
        written.append(path)
        log(f"    wrote {os.path.basename(path)}")
    return written

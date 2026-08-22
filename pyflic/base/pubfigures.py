"""Publication Figures: plotnine-rendered, journal-ready vector figures.

A separate rendering path from the matplotlib figures embedded in the PDF
report: the two share the same summarized, exclusion-filtered data — never
rendering code.  A figure is defined by a per-plot :class:`PlotSpec` (content)
plus a named, reusable :class:`PlotStyle` (look).  Both persist in
``<project>/plot_specs.yaml``, written by the Plot Editor; saved figures land in
``<project>/figures/``.

pyflic has **two** spec families where PyTrackingAnalysis has one:

* ``faceted_<metric>`` — x = Treatment, one panel per Facet, jittered points
  with a mean overlay.  Expressible only because a time window is now a column
  (ADR-0008).
* ``timecourse_<metric>`` — x = time bin, one line per Treatment with an SEM
  ribbon.  The figure that most distinguishes FLIC data, and the reason the
  Editor carries two preview forms.

Both share the ``styles:`` block, so a Project has one look.

SVG output uses ``svg.fonttype='none'`` so labels arrive in Illustrator as live,
editable text; PDF embeds TrueType (fonttype 42) for the same reason.
"""

from __future__ import annotations

import io
import os
from dataclasses import asdict, dataclass, field

import matplotlib
import pandas as pd
import yaml

SPECS_FILENAME = "plot_specs.yaml"
FIGURES_DIRNAME = "figures"

#: Ordered fallback palette for treatments a style does not name explicitly.
DEFAULT_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#d97706",
                   "#7c3aed", "#0891b2", "#64748b", "#be185d"]

_THEMES = ("classic", "bw", "minimal")
_MEAN_STYLES = ("point+sem", "bar+sem", "point+95ci", "bar+95ci")
_GEOMS = ("dots", "box", "box+dots")
_STRIP_STYLES = ("plain", "boxed")

FAMILY_FACETED = "faceted"
FAMILY_TIMECOURSE = "timecourse"

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
}


def family_of(plot_id: str) -> str:
    return PLOT_TYPES.get(plot_id, {}).get("family", FAMILY_FACETED)


def plots_for_layout(chamber_layout: str) -> list[str]:
    """Plot ids valid for *chamber_layout* — a PI figure is meaningless on a
    single-well recording, so the Editor never offers one."""
    out = []
    for plot_id, info in PLOT_TYPES.items():
        needed = info.get("layout")
        if needed is None or needed == chamber_layout:
            out.append(plot_id)
    return out


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
    #: treatment name -> {"label": display, "show": bool}; dict order = plot order.
    treatments: dict = field(default_factory=dict)
    y_limits: list | None = None
    ref_line: float | None = None
    free_y: bool = False
    #: Give each Replicate its own point shape so batch structure is visible
    #: in the pooled figure.  Ignored when the data has no Experiment column.
    mark_experiments: bool = False
    #: Time-course family: bin width in minutes, and whether to draw the SEM
    #: ribbon around each treatment's mean.
    binsize: float = 30.0
    ribbon: bool = True

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
        return spec


def default_spec(plot_id: str, well_a: str = "well A") -> PlotSpec:
    info = PLOT_TYPES[plot_id]
    y_limits = info["y_limits"]
    return PlotSpec(
        y_label=str(info["y_label"]).replace("well A", well_a),
        y_limits=list(y_limits) if y_limits is not None else None,
        ref_line=info["ref_line"],
        free_y=bool(info.get("free_y", False)),
        x_label="" if info["family"] == FAMILY_FACETED else "Time (min)",
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

def _metric_values(df: pd.DataFrame, metric: str) -> pd.Series:
    from .analytics import _resolve_metric_col

    return pd.to_numeric(_resolve_metric_col(df, metric), errors="coerce")


def faceted_data(frame: pd.DataFrame, metric: str,
                 label_order: list[str] | None = None) -> pd.DataFrame:
    """Tidy per-chamber data for one metric: Treatment, Phase, Value [, Experiment].

    *frame* is a faceted summary (a Replicate's ``feeding_summary_facet.csv`` or
    a Project's ``_Summary_Facet.csv``).  An unfaceted frame becomes a single
    "Whole recording" phase, so the same builder serves both.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Treatment", "Phase", "Value"])
    if "Facet" in frame.columns:
        phase = frame["Facet"].astype(str)
        order = label_order or list(dict.fromkeys(phase))
    else:
        phase = pd.Series(["Whole recording"] * len(frame), index=frame.index)
        order = ["Whole recording"]
    out = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "Phase": pd.Categorical(phase, categories=order, ordered=True),
        "Value": _metric_values(frame, metric),
    })
    if "Experiment" in frame.columns:
        out["Experiment"] = frame["Experiment"].astype(str)
    out = out[(out["Treatment"] != "") & out["Value"].notna()]
    return out.reset_index(drop=True)


def timecourse_data(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tidy per-chamber binned data: Treatment, Minutes, Value [, Experiment].

    *frame* is a binned summary (``binned_feeding_summary.csv``), which carries
    one row per chamber per time bin with a ``Minutes`` bin-midpoint column.
    """
    if frame is None or frame.empty or "Minutes" not in frame.columns:
        return pd.DataFrame(columns=["Treatment", "Minutes", "Value"])
    out = pd.DataFrame({
        "Treatment": frame["Treatment"].astype(str).str.strip(),
        "Minutes": pd.to_numeric(frame["Minutes"], errors="coerce"),
        "Value": _metric_values(frame, metric),
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


def resolve_font_family(requested: str) -> list[str]:
    """*requested* first, with generic fallbacks, so a missing face degrades to
    something sane instead of matplotlib's default at some sizes only."""
    wanted = str(requested or "").strip()
    fallbacks = ["DejaVu Sans", "Liberation Sans", "sans-serif"]
    return ([wanted] + fallbacks) if wanted else fallbacks


def effective_width_mm(style: PlotStyle, n_facets: int | None = None) -> float:
    if style.facet_width_mm and style.facet_width_mm > 0 and n_facets:
        return _AXIS_MARGIN_MM + float(style.facet_width_mm) * int(n_facets)
    return float(style.width_mm)


def effective_height_mm(style: PlotStyle, spec: PlotSpec | None = None) -> float:
    if not (style.facet_height_mm and style.facet_height_mm > 0):
        return float(style.height_mm)
    line_mm = float(style.base_pt) * _PT_TO_MM
    margin = 10.0 + 2.5 * line_mm
    if spec is not None:
        if (spec.x_label or "").strip():
            margin += 1.8 * line_mm
        if (spec.title or "").strip():
            margin += 1.8 * (line_mm + 2 * _PT_TO_MM)
    return float(style.facet_height_mm) + margin


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
    """Per-treatment jittered points with a mean overlay, one panel per Facet."""
    import numpy as np
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
    all_phases = list(data["Phase"].cat.categories)
    include = [p for p in (spec.facets or all_phases) if p in all_phases]
    data = data[data["Phase"].isin(include)].copy()
    shown = [str(spec.facet_labels.get(p, p)) for p in include]
    data["Phase"] = pd.Categorical(
        data["Phase"].astype(str).map(lambda p: str(spec.facet_labels.get(p, p))),
        categories=shown, ordered=True)

    mark = bool(spec.mark_experiments) and "Experiment" in data.columns
    g = (p9.ggplot(data, p9.aes("Treatment", "Value", color="Treatment"))
         + p9.facet_wrap("~Phase", nrow=1,
                         scales="free_y" if spec.free_y else "fixed")
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
        ## Only the shape (replicate) legend is useful — treatments are named
        ## on the x axis already.
        g = g + p9.guides(color="none", fill="none")

    if not boxed:
        stat = data.groupby(["Phase", "Treatment"], observed=True)["Value"].agg(
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
    return g + _theme_for(style, n_facets=len(shown),
                          height_mm=effective_height_mm(style, spec),
                          show_legend=mark)


def build_timecourse(df: pd.DataFrame, spec: PlotSpec, style: PlotStyle):
    """One line per treatment over time bins, with an SEM ribbon."""
    import plotnine as p9

    data, labels, colors = _apply_treatment_order(df, spec, style)
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


def build_figure(plot_id: str, df: pd.DataFrame, spec: PlotSpec,
                 style: PlotStyle):
    """Dispatch to the builder for *plot_id*'s family."""
    if family_of(plot_id) == FAMILY_TIMECOURSE:
        return build_timecourse(df, spec, style)
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

def project_frames(project):
    """``(facet_frame, binned_frame)`` for a Project's pooled figures.

    The facet frame comes from the Combined Analysis; the binned frame is
    stacked here from each Replicate's saved ``binned_feeding_summary.csv``,
    because a time course needs bins the faceted summary does not carry.
    """
    facet_path = os.path.join(project.analysis_path,
                              f"{project.name}_Summary_Facet.csv")
    facet = pd.read_csv(facet_path) if os.path.isfile(facet_path) else None
    if facet is None:
        summary_path = os.path.join(project.analysis_path,
                                    f"{project.name}_Summary.csv")
        facet = (pd.read_csv(summary_path)
                 if os.path.isfile(summary_path) else None)

    binned_frames = []
    for name in project.experiment_names:
        path = os.path.join(project.experiment_dir(name), "analysis",
                            "binned_feeding_summary.csv")
        if os.path.isfile(path):
            df = pd.read_csv(path)
            df.insert(0, "Experiment", name)
            binned_frames.append(df)
    binned = pd.concat(binned_frames, ignore_index=True) if binned_frames else None
    return facet, binned


def render_all(project, fmt: str = "svg", out_dir: str | None = None,
               only: list[str] | None = None, log=print) -> list[str]:
    """Render every Plot Spec in the Project's ``plot_specs.yaml``.

    A plot whose data is unavailable (a time course with no saved binned
    summaries, a PI figure on a single-well Project) is skipped with a log
    line rather than failing the run — one missing figure must not abort an
    unattended Batch Run.
    """
    specs = load_project_specs(project.project_directory)
    facet, binned = project_frames(project)
    target = out_dir or os.path.join(project.project_directory, FIGURES_DIRNAME)
    os.makedirs(target, exist_ok=True)

    _, label_of = (project.window_labels(facet)
                   if facet is not None and "FacetRange" in facet.columns
                   else (None, None))
    label_order = None
    if label_of:
        label_order = list(dict.fromkeys(label_of.values()))

    written: list[str] = []
    for plot_id, spec in specs.plots.items():
        if only and plot_id not in only:
            continue
        info = PLOT_TYPES[plot_id]
        metric = info["metric"]
        family = info["family"]
        source = binned if family == FAMILY_TIMECOURSE else facet
        if source is None or source.empty:
            log(f"    skipped {plot_id}: no "
                f"{'binned' if family == FAMILY_TIMECOURSE else 'summary'} data")
            continue
        df = (timecourse_data(source, metric) if family == FAMILY_TIMECOURSE
              else faceted_data(source, metric, label_order))
        if df.empty:
            log(f"    skipped {plot_id}: metric '{metric}' not in the data")
            continue
        try:
            g = build_figure(plot_id, df, spec, specs.style_for(spec))
            path = save_figure(g, os.path.join(target, f"{plot_id}.{fmt}"), fmt)
        except Exception as err:  # noqa: BLE001
            log(f"    skipped {plot_id}: {type(err).__name__}: {err}")
            continue
        written.append(path)
        log(f"    wrote {os.path.basename(path)}")
    return written

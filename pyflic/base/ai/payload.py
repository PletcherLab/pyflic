"""Build the provider-agnostic input for an AI Summary.

The AI is given exactly what the report's reader sees: the analysis text, the
figures as PNGs, and the small per-chamber summary CSVs — never the rendered
PDF and never raw per-sample signal data.  It *summarizes* the pipeline's
analysis; it does not perform its own.
"""

from __future__ import annotations

import io
import os

from .base import SummaryPayload

# Guardrails, not tuning knobs. They only bite on a pathological project, where
# the note they leave in the text tells the model — and anyone reading a prompt
# dump — that content was dropped.
MAX_IMAGES = 16
MAX_CSV_CHARS = 100_000

PROJECT_INSTRUCTIONS = (
    "You are writing a one-page summary of a pooled analysis of FLIC "
    "(Fly Liquid-food Interaction Counter) feeding experiments, for the "
    "scientist who ran them.\n\n"
    "You are given the report's own content: its cover text, its pooled "
    "figures, its statistics table, and the per-chamber summary CSVs.\n\n"
    "Rules:\n"
    "- Summarize what the analysis found. Do NOT perform your own analysis, "
    "recompute anything, or introduce statistics that are not in the input.\n"
    "- Every number you state must appear in the input.\n"
    "- Say plainly when a comparison is not significant. Do not imply a trend "
    "the statistics do not support.\n"
    "- Note where the pooled and mixed-model p-values disagree: that gap is "
    "between-replicate variation, and it matters.\n"
    "- Plain prose, no headings, at most 400 words."
)


def _read_csv_text(path: str, budget: int) -> tuple[str, int]:
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read(budget + 1)
    except OSError:
        return "", budget
    if len(text) > budget:
        return (text[:budget] + "\n[...truncated...]"), 0
    return text, budget - len(text)


def build_project_payload(project) -> SummaryPayload:
    """Serialize a Project's Combined Analysis for the AI.

    Figures are rendered fresh from the same Spec+Style the report uses, so the
    AI is looking at the figures the reader will see rather than at a stale
    ``figures/`` directory.
    """
    from .. import pubfigures
    from ..project_report import cover_text

    summary, facet, missing = project.combined_frames()
    if summary is None:
        raise ValueError(
            f"Project '{project.name}' has no combined analysis to summarize.")

    lines: list[str] = [f"# Project report — {project.name}", "",
                        cover_text(project, summary, missing), "",
                        "=== Statistics ===",
                        project.stats_text(summary, facet)]

    images: list[tuple[str, bytes]] = []
    dropped = 0
    specs = pubfigures.load_project_specs(project.project_directory)
    _facet, binned = pubfigures.project_frames(project)
    for plot_id in project.experiment_type.report_set(project.chamber_layout):
        if plot_id not in pubfigures.PLOT_TYPES:
            continue
        info = pubfigures.PLOT_TYPES[plot_id]
        needed = info.get("layout")
        if needed is not None and needed != project.chamber_layout:
            continue
        source = binned if info["family"] == pubfigures.FAMILY_TIMECOURSE else facet
        if source is None or source.empty:
            continue
        df = (pubfigures.timecourse_data(source, info["metric"])
              if info["family"] == pubfigures.FAMILY_TIMECOURSE
              else pubfigures.faceted_data(source, info["metric"]))
        if df.empty:
            continue
        if len(images) >= MAX_IMAGES:
            dropped += 1
            continue
        spec = specs.plots.get(plot_id) or pubfigures.default_spec(plot_id)
        try:
            g = pubfigures.build_figure(plot_id, df, spec, specs.style_for(spec))
            buffer = io.BytesIO()
            g.save(buffer, format="png", dpi=110, verbose=False)
            images.append((info["display"], buffer.getvalue()))
        except Exception:  # noqa: BLE001
            dropped += 1
    if dropped:
        lines.append(f"\n[{dropped} figure(s) omitted from this payload.]")

    budget = MAX_CSV_CHARS
    for suffix in ("_Summary.csv", "_Summary_Facet.csv", "_Excluded.csv"):
        path = os.path.join(project.analysis_path, f"{project.name}{suffix}")
        if not os.path.isfile(path) or budget <= 0:
            continue
        text, budget = _read_csv_text(path, budget)
        if text:
            lines.append(f"\n=== {os.path.basename(path)} ===\n{text}")

    return SummaryPayload(text="\n".join(lines), images=images)

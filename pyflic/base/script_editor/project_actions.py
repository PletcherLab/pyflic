"""Action catalogue for **Project Scripts** (ADR-0005/0006).

A Project Script is a saved step list of project-level actions, stored in
``project.yaml`` under ``scripts:``.  It has the same shape and the same visual
editor as an Experiment Script but a **separate action registry** — the levels
cannot mix.  The only bridge is ``run_in_experiments``, which runs a named
Experiment Script in every Replicate.

There is no third (Batch) script level: what a Batch Run executes IS a Project
Script, named by ``batch.yaml``'s ``script:`` key.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from ..ui import Category

ParamType = Literal["choice", "bool", "string", "int", "float", "list_str"]


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    type: ParamType
    note: str = ""
    default: Any = None
    choices: list[str] | None = None
    required: bool = False


@dataclass(frozen=True)
class ProjectAction:
    action: str
    label: str
    blurb: str
    icon: str
    category: Category
    produces: Literal["figure", "csv", "pdf", "none"]
    params: list[Param] = field(default_factory=list)
    notes: str = ""


PROJECT_ACTIONS: list[ProjectAction] = [
    ProjectAction(
        action="validate_design",
        label="Validate design",
        blurb="Check every replicate against the project design",
        icon="basic", category=Category.LOAD, produces="none",
        notes="Fails the script when a replicate contradicts the design. "
              "Loading the Project already performs this check, so a script "
              "only needs the step to fail *early*, before an expensive run.",
    ),
    ProjectAction(
        action="run_in_experiments",
        label="Run script in replicates",
        blurb="The only bridge from project level to experiment level",
        icon="scripts", category=Category.SCRIPTS, produces="none",
        params=[
            Param(key="script", label="Experiment Script name", type="string",
                  note="Resolved project-first: the Project's "
                       "experiment_scripts: section, then each replicate's own "
                       "scripts:.",
                  required=True),
            Param(key="only", label="Only these replicates", type="list_str",
                  note="Directory names, comma separated. Blank = every "
                       "replicate."),
        ],
        notes="Continue-on-error: a replicate where the name resolves nowhere "
              "is logged and counted, and the run carries on.",
    ),
    ProjectAction(
        action="run_all_analyses",
        label="Analyze all replicates",
        blurb="Basic analysis in every replicate",
        icon="basic", category=Category.ANALYZE, produces="csv",
        params=[
            Param(key="skip_analyzed", label="Skip already-analyzed", type="bool",
                  default=False,
                  note="Skip replicates that already have a saved "
                       "feeding_summary.csv."),
            Param(key="reports", label="Also write per-replicate reports",
                  type="bool", default=False),
        ],
    ),
    ProjectAction(
        action="build_combined_analysis",
        label="Build combined analysis",
        blurb="Stack replicate summaries; pooled + mixed statistics",
        icon="csv", category=Category.ANALYZE, produces="csv",
        notes="Replicates with no saved summary are omitted and listed — "
              "never silently analyzed.",
    ),
    ProjectAction(
        action="project_report",
        label="Create project report",
        blurb="Pooled figures, statistics, per-replicate table",
        icon="pdf", category=Category.ANALYZE, produces="pdf",
        params=[
            Param(key="ai_summary", label="Include AI narrative", type="bool",
                  default=False,
                  note="Requires a provider API key in .env."),
        ],
        notes="Builds the Combined Analysis first if it is missing, so this "
              "one action is the whole Create-report button.",
    ),
    ProjectAction(
        action="render_publication_figures",
        label="Render publication figures",
        blurb="Vector figures from plot_specs.yaml into figures/",
        icon="plot", category=Category.PLOTS, produces="figure",
        params=[
            Param(key="format", label="Format", type="choice",
                  choices=["svg", "pdf"], default="svg"),
            Param(key="only", label="Only these plot ids", type="list_str",
                  note="Blank = every plot in plot_specs.yaml."),
        ],
        notes="Skipped with a log line when the Project has no "
              "plot_specs.yaml — an unattended run must not fail for want of "
              "a curated figure set.",
    ),
    ProjectAction(
        action="generate_ai_narrative",
        label="Generate AI narrative",
        blurb="AI-written summary of the combined analysis",
        icon="scripts", category=Category.TOOLS, produces="none",
        params=[
            Param(key="provider", label="Provider", type="choice",
                  choices=["anthropic", "openai"], default="anthropic"),
        ],
    ),
]

PROJECT_ACTIONS_BY_NAME: dict[str, ProjectAction] = {
    a.action: a for a in PROJECT_ACTIONS
}

# ---------------------------------------------------------------------------
# Built-in pipelines
# ---------------------------------------------------------------------------

#: Every Project can run these without authoring anything.  Never written to
#: ``project.yaml``, so they track the shipped default.
BUILTIN_PROJECT_SCRIPTS: dict[str, dict] = {
    "Standard Pipeline": {
        "name": "Standard Pipeline",
        "steps": [
            {"action": "validate_design"},
            {"action": "project_report"},
            {"action": "render_publication_figures"},
        ],
    },
    "Report Pipeline": {
        "name": "Report Pipeline",
        "steps": [
            {"action": "project_report"},
            {"action": "render_publication_figures"},
        ],
    },
}


def builtin_project_script(name: str) -> dict | None:
    """A built-in pipeline by name, or ``None``."""
    script = BUILTIN_PROJECT_SCRIPTS.get(name)
    return dict(script) if script else None


def default_project_script() -> dict:
    """The Project Script written into every new ``project.yaml``.

    Named ``batch`` for the job it does rather than the built-in it was seeded
    from, so a reader of the yaml can see what a Batch Run will do there.  It
    holds the Report Pipeline's steps — preferred over the Standard Pipeline
    for an unattended run because it does not gate on ``validate_design``,
    which would fail Projects mid-migration.
    """
    return {
        "name": "batch",
        "steps": [
            {"action": "project_report"},
            {"action": "render_publication_figures"},
        ],
    }

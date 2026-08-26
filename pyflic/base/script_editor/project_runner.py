"""Executor for **Project Scripts**.

Mirrors the Experiment-level runner one level up.  Every action here is a
project-level action; the only step that reaches down a level is
``run_in_experiments``, which is the sole bridge between the two registries
(ADR-0005/0006).
"""

from __future__ import annotations

import os

from .project_actions import PROJECT_ACTIONS_BY_NAME


def _as_list(value) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def validate_project_script(project, script: dict) -> list[str]:
    """Problems with *script* before running it, as human-readable lines.

    Pre-run validation exists so an overnight Batch Run fails on a typo in the
    first second rather than after the third member's analysis.
    """
    problems: list[str] = []
    for index, step in enumerate(script.get("steps") or [], start=1):
        action = str((step or {}).get("action") or "")
        if action not in PROJECT_ACTIONS_BY_NAME:
            problems.append(
                f"step {index}: unknown project action '{action}'"
                + (" — that is an experiment-level action; run it through "
                   "run_in_experiments" if action in _experiment_action_names()
                   else ""))
            continue
        if action == "run_in_experiments":
            name = str(step.get("script") or "").strip()
            if not name:
                problems.append(f"step {index}: run_in_experiments needs a "
                                f"'script' name")
            elif project is not None:
                resolvable = project.find_experiment_script(name) is not None
                if not resolvable:
                    resolvable = any(
                        _member_script(project, rep, name) is not None
                        for rep in project.member_names)
                if not resolvable:
                    problems.append(
                        f"step {index}: no Experiment Script named '{name}' in "
                        f"the project's experiment_scripts: or in any "
                        f"member's scripts:")
            for only in _as_list(step.get("only")):
                if project is not None and only not in project.member_names:
                    problems.append(
                        f"step {index}: 'only' names member '{only}', which "
                        f"is not in this project")
    return problems


def _experiment_action_names() -> set[str]:
    from .actions import ACTIONS

    return {a.action for a in ACTIONS}


def _member_script(project, member: str, name: str) -> dict | None:
    for script in (project.configs.get(member, {}).get("scripts") or []):
        if isinstance(script, dict) and script.get("name") == name:
            return script
    return None


def run_project_script(project, script: dict, log=print) -> dict:
    """Run *script* against *project*.  Returns a per-step outcome summary."""
    problems = validate_project_script(project, script)
    if problems:
        raise ValueError("Project Script '"
                         + str(script.get("name") or "?")
                         + "' is not runnable:\n  - " + "\n  - ".join(problems))

    outcomes: list[dict] = []
    for index, step in enumerate(script.get("steps") or [], start=1):
        action = str(step.get("action"))
        label = PROJECT_ACTIONS_BY_NAME[action].label
        log(f"[{index}] {label}")
        try:
            _dispatch(project, action, step, log)
            outcomes.append({"step": index, "action": action, "ok": True})
        except Exception as err:  # noqa: BLE001
            outcomes.append({"step": index, "action": action, "ok": False,
                             "error": f"{type(err).__name__}: {err}"})
            raise
    return {"script": script.get("name"), "steps": outcomes}


def _dispatch(project, action: str, step: dict, log) -> None:
    if action == "validate_design":
        ## Loading the Project already validated; re-reading it here is what
        ## makes the step fail *early* if the yaml changed under a long run.
        from ..project import Project

        Project(project.project_directory)
        log(f"    design OK — {len(project.member_names)} member(s)")
        for warning in project.warnings:
            log(f"    note: {warning}")
        return

    if action == "run_in_experiments":
        name = str(step.get("script")).strip()
        only = _as_list(step.get("only"))
        targets = only or list(project.member_names)
        central = project.find_experiment_script(name)
        failures = 0
        for member in targets:
            recipe = central or _member_script(project, member, name)
            if recipe is None:
                log(f"    [{member}] no script '{name}' — skipped")
                failures += 1
                continue
            try:
                _run_experiment_script(project, member, recipe, log)
            except Exception as err:  # noqa: BLE001
                failures += 1
                log(f"    [{member}] FAILED: {type(err).__name__}: {err}")
        log(f"    {len(targets) - failures}/{len(targets)} member(s) ok")
        return

    if action == "run_all_analyses":
        failures = project.run_all(
            make_reports=bool(step.get("reports", False)),
            skip_analyzed=bool(step.get("skip_analyzed", False)),
            log=lambda m: log(f"    {m}"))
        if failures:
            raise RuntimeError(
                f"{len(failures)} member(s) failed: " + "; ".join(failures))
        return

    if action == "build_combined_analysis":
        result = project.build_combined_analysis()
        for path in result["written"]:
            log(f"    wrote {os.path.basename(path)}")
        if result["missing"]:
            log(f"    omitted (no saved analysis): "
                f"{', '.join(result['missing'])}")
        return

    if action == "project_report":
        from ..project_report import write_project_report

        path = write_project_report(
            project, ai_summary=bool(step.get("ai_summary", False)), log=log)
        log(f"    wrote {os.path.basename(path)}")
        return

    if action == "render_publication_figures":
        from .. import pubfigures

        specs_path = os.path.join(project.project_directory,
                                  pubfigures.SPECS_FILENAME)
        if not os.path.isfile(specs_path):
            ## Never a failure: an unattended Batch Run must not stop because
            ## a Project has no curated figure set yet.
            log("    skipped — no plot_specs.yaml in this project")
            return
        written = pubfigures.render_all(
            project, fmt=str(step.get("format") or "svg"),
            only=_as_list(step.get("only")) or None, log=log)
        log(f"    {len(written)} figure(s) written")
        return

    if action == "generate_ai_narrative":
        from ..ai import generate_project_narrative

        path = generate_project_narrative(
            project, provider=str(step.get("provider") or "anthropic"))
        log(f"    wrote {os.path.basename(path)}")
        return

    raise ValueError(f"Unhandled project action '{action}'")


def _run_experiment_script(project, member: str, recipe: dict, log) -> None:
    """Run one Experiment Script inside one Member.

    Goes through the Project so the Member loads with the Design's
    ``global:`` inherited and per-DFM overrides restricted.
    """
    from .runner import run_experiment_script

    log(f"    [{member}] {recipe.get('name')}")
    exp = project.load_member(member)
    run_experiment_script(exp, recipe, log=lambda m: log(f"      [{member}] {m}"))

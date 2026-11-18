"""
Schema linter for ``flic_config.yaml``.

Reports problems with line/column information when possible.  Use as:

    issues = lint_flic_config(Path("project/flic_config.yaml"))
    for i in issues:
        print(i.format())
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml

from .utils import ENVIRONMENTAL_MONITOR_MESSAGE, is_environmental_monitor


_KNOWN_GLOBAL_KEYS = {
    "params", "parameters", "well_names", "constants",
    "experimental_design_factors", "experiment_type",
    # Post-overhaul keys (ADR-0007 / ADR-0008).
    "chamber_layout", "facet_cutoffs", "facet_labels", "transform_licks",
    "exclusion_group",
    # The optogenetic light QC's switch (auto | yes | no).
    "optogenetics",
    # Where the Facets come from: cutoffs | program | program_cycles.
    "facet_source",
    # The Selection: what Run QC / Run analysis / Generate plots leave out (ADR-0018).
    "omit",
    # The Treatment control's level, under a type with one (ADR-0019).
    "control_level",
    # Slices: separate analyses of part of the recording (ADR-0020).
    "slices", "also_exclude", "slice_tolerance_min",
}
_KNOWN_PARAM_KEYS = {
    "baseline_window_minutes", "baseline_window_min", "baseline_window",
    "feeding_threshold", "feeding_minimum", "feeding_minevents",
    "feeding_event_link_gap", "link_gap",
    "tasting_minimum", "tasting_maximum", "tasting_low", "tasting_high",
    "tasting_minevents",
    "samples_per_second", "samples_per_sec",
    "chamber_size", "chamber_sets",
    "correct_for_dual_feeding", "pi_direction", "pi_multiplier",
}
_KNOWN_DFM_KEYS = {
    "id", "ID", "params", "parameters", "chambers", "Chambers",
    "excluded_chambers", "well_names", "paired_chambers", "optogenetics",
    # Progressive Ratio's Chamber Groups without Program.txt: paired | single.
    "chamber_groups",
}


@dataclass(frozen=True, slots=True)
class LintIssue:
    severity: str   # "error" | "warning"
    message: str
    line: int | None = None
    column: int | None = None
    path: str | None = None

    def format(self, file: Path | None = None) -> str:
        loc = ""
        if file is not None:
            loc = str(file)
        if self.line is not None:
            loc += f":{self.line}"
            if self.column is not None:
                loc += f":{self.column}"
        prefix = f"{loc} " if loc else ""
        return f"{prefix}{self.severity}: {self.message}"


def _emit(
    issues: list[LintIssue],
    severity: str,
    message: str,
    node: Any | None = None,
    path: str | None = None,
) -> None:
    line = column = None
    if node is not None and hasattr(node, "start_mark"):
        line = node.start_mark.line + 1
        column = node.start_mark.column + 1
    issues.append(LintIssue(severity, message, line, column, path))


def _lint_type_definition(issues: list[LintIssue], path: Path, g: dict, program) -> None:
    """The Experiment Type as ADR-0019 keeps it: a lab type's copy beside
    the data and its agreement with the library, and the light it expects
    against the Opto Program (which the loader checks too, unless
    ``optogenetics: no`` keeps it from reading the program)."""
    from . import opto_program as _opto
    from .experiment_types import COPY_NAME, get_experiment_type, library

    try:
        etype = get_experiment_type(g.get("experiment_type"), path.parent)
    except ValueError:
        return  # the unknown type is reported by _resolved_layout
    if etype.resolved_from == "library":
        _emit(issues, "warning",
              f"experiment_type '{etype.name}' is read from the lab library and the "
              f"data carries no copy of it: save the design or the config in the "
              f"editor to write {COPY_NAME} beside it, so another machine analyses "
              f"it the same way", path="global.experiment_type")
    drift = library.copy_drift(path.parent)
    if drift:
        _emit(issues, "warning", drift, path="global.experiment_type")
    if program is None:
        return
    try:
        setting = _opto.normalize_setting(g.get(_opto.SETTING_KEY))
    except ValueError:
        setting = _opto.SETTING_AUTO
    if setting == _opto.SETTING_NO:
        return
    for problem in etype.light_problems(program):
        if etype.name == "Hedonic":
            problem += (" (Hedonic expects no light: build a hedonic type that expects "
                        "this light in the Experiment Type Editor)")
        _emit(issues, "error", problem, path="global.experiment_type")


def lint_flic_config(path: str | Path) -> list[LintIssue]:
    """Validate *path* against the flic_config.yaml schema.

    Returns a list of issues; an empty list means the file is well-formed.
    """
    path = Path(path)
    issues: list[LintIssue] = []

    if not path.is_file():
        issues.append(LintIssue("error", f"file does not exist: {path}"))
        return issues

    text = path.read_text(encoding="utf-8")
    try:
        # Use compose to keep node line numbers
        node = yaml.compose(text)
    except yaml.YAMLError as exc:
        line = column = None
        if hasattr(exc, "problem_mark") and exc.problem_mark is not None:
            line = exc.problem_mark.line + 1
            column = exc.problem_mark.column + 1
        issues.append(LintIssue("error", f"YAML parse error: {exc.problem}", line, column))
        return issues

    if node is None:
        issues.append(LintIssue("error", "file is empty"))
        return issues

    cfg = yaml.safe_load(text)
    if not isinstance(cfg, dict):
        _emit(issues, "error", "root must be a mapping", node)
        return issues

    # Top-level keys
    from .layout import DATA_FOLDER_KEY, configured_data_folder

    expected_top = {"global", "dfms", "DFMs", "scripts", DATA_FOLDER_KEY}
    for k in cfg.keys():
        if k == "data_dir":
            ## Word for word as it has read since data_dir was retired, unless
            ## the config also names data_folder: (ADR-0021), where "always
            ## data/" would be untrue.  A blank data_folder: is no key.
            if configured_data_folder(path.parent, cfg) is None:
                where = "data is always read from experiment_dir/data/"
            else:
                where = f"data is read from the folder {DATA_FOLDER_KEY}: names"
            _emit(
                issues,
                "warning",
                f"'data_dir' is no longer used and should be removed; {where}",
                path="data_dir",
            )
        elif k not in expected_top:
            _emit(issues, "warning", f"unknown top-level key {k!r}", path=k)
    ## Where the loader will read the recording (ADR-0021): its problems are
    ## this config's, and every check below that reads the recording reads it
    ## there.  None when the loader would refuse the location.
    data_folder = _lint_data_folder(issues, path, cfg)

    g = cfg.get("global") or {}
    if g and not isinstance(g, dict):
        _emit(issues, "error", "global: must be a mapping", path="global")
        g = {}
    for k in g.keys():
        if k == DATA_FOLDER_KEY:
            _emit(issues, "error",
                  f"{DATA_FOLDER_KEY} belongs at the top of {path.name}, never under "
                  f"global: — there it is not read, and the recording is read from "
                  f"data/", path=f"global.{k}")
        elif k not in _KNOWN_GLOBAL_KEYS:
            _emit(issues, "warning", f"unknown global key {k!r}", path=f"global.{k}")
    _lint_omit(issues, path, g)
    _lint_slices(issues, path, g)

    g_params = g.get("params") or g.get("parameters") or {}
    chamber_size_global = None
    if isinstance(g_params, dict):
        for k in g_params.keys():
            if k not in _KNOWN_PARAM_KEYS:
                _emit(
                    issues, "warning",
                    f"unknown param {k!r} (will be ignored by loader)",
                    path=f"global.params.{k}",
                )
        chamber_size_global = g_params.get("chamber_size")
        pi = g_params.get("pi_direction")
        if pi is not None and str(pi).lower() not in ("left", "right"):
            _emit(issues, "error",
                  f"global.params.pi_direction must be 'left' or 'right', got {pi!r}",
                  path="global.params.pi_direction")

    ## The optogenetic light QC: its switch, its constants, and the Opto
    ## Program in data/ — which the loader reads, so its problems are this
    ## config's problems too.
    from . import opto_program as _opto
    from .opto_qc import opto_constant_problems

    problem = _opto.setting_problem(g.get(_opto.SETTING_KEY),
                                    where=f"global.{_opto.SETTING_KEY}")
    if problem:
        _emit(issues, "error", problem, path=f"global.{_opto.SETTING_KEY}")
    for problem in opto_constant_problems(g.get("constants")):
        _emit(issues, "error", problem, path="global.constants")
    program = None
    try:
        program_path = (_opto.find_program(data_folder)
                        if data_folder is not None else None)
    except _opto.ProgramError as err:
        _emit(issues, "error", str(err))
        program_path = None
    if program_path is not None:
        try:
            program = _opto.read_program(program_path)
        except _opto.ProgramError as err:
            _emit(issues, "warning", f"{program_path.name} will not be used: {err}")
        else:
            for dfm_id, reason in sorted(program.rejected.items()):
                _emit(issues, "warning", f"{program_path.name}: DFM {dfm_id} section "
                                         f"dropped — {reason}")
    _lint_type_definition(issues, path, g, program)

    factors = g.get("experimental_design_factors") or {}
    factor_names = list(factors.keys()) if isinstance(factors, dict) else []
    factor_levels: dict[str, list[str]] = {}
    if factor_names:
        for fname, lvls in factors.items():
            if not isinstance(lvls, list) or not lvls:
                _emit(
                    issues, "error",
                    f"factor {fname!r}: levels must be a non-empty list",
                    path=f"global.experimental_design_factors.{fname}",
                )
                factor_levels[fname] = []
            else:
                factor_levels[fname] = [str(x) for x in lvls]

    dfms_node = cfg.get("dfms", cfg.get("DFMs"))
    if dfms_node is None:
        _emit(issues, "error", "missing required 'dfms' section")
        return issues

    if isinstance(dfms_node, dict):
        dfm_items = list(dfms_node.items())
    elif isinstance(dfms_node, list):
        dfm_items = [(node.get("id", node.get("ID")), node) for node in dfms_node if isinstance(node, dict)]
    else:
        _emit(issues, "error", "dfms must be a mapping or list", path="dfms")
        return issues

    _lint_facet_source(issues, path, g, dfm_items, program, program_path, data_folder)
    layout = _resolved_layout(issues, path, g)

    seen_ids: set[int] = set()
    for raw_id, dfm in dfm_items:
        if not isinstance(dfm, dict):
            _emit(issues, "error", f"DFM {raw_id!r}: entry must be a mapping")
            continue
        try:
            dfm_id = int(raw_id if raw_id is not None else dfm.get("id"))
        except (TypeError, ValueError):
            _emit(issues, "error", f"DFM id is not an integer: {raw_id!r}")
            continue
        if dfm_id in seen_ids:
            _emit(issues, "error", f"DFM {dfm_id}: duplicate id")
        seen_ids.add(dfm_id)
        if is_environmental_monitor(dfm_id):
            ## The loader refuses it with the same words; nothing else about
            ## the entry matters.
            _emit(issues, "error", f"{ENVIRONMENTAL_MONITOR_MESSAGE}; remove id "
                                   f"{dfm_id} from dfms:", path=f"dfms.{dfm_id}")
            continue

        for k in dfm.keys():
            if k not in _KNOWN_DFM_KEYS:
                _emit(issues, "warning", f"DFM {dfm_id}: unknown key {k!r}",
                      path=f"dfms.{dfm_id}.{k}")

        d_params = dfm.get("params") or dfm.get("parameters") or {}
        if isinstance(d_params, dict):
            for k in d_params.keys():
                if k not in _KNOWN_PARAM_KEYS:
                    _emit(issues, "warning",
                          f"DFM {dfm_id}: unknown param {k!r}",
                          path=f"dfms.{dfm_id}.params.{k}")
            pi = d_params.get("pi_direction")
            if pi is not None and str(pi).lower() not in ("left", "right"):
                _emit(issues, "error",
                      f"DFM {dfm_id}: pi_direction must be 'left' or 'right', got {pi!r}",
                      path=f"dfms.{dfm_id}.params.pi_direction")

        ## The loader never reads chamber_size: it derives it from the layout
        ## (ADR-0007).  A stated one that disagrees is silently overruled.
        stated = (d_params.get("chamber_size") if isinstance(d_params, dict) else None)
        if stated is None:
            stated = chamber_size_global
        if layout is not None and stated is not None:
            from .experiment_types import LAYOUT_CHAMBER_SIZE

            wells = LAYOUT_CHAMBER_SIZE[layout]
            try:
                agrees = int(stated) == wells
            except (TypeError, ValueError):
                agrees = False
            if not agrees:
                _emit(issues, "warning",
                      f"DFM {dfm_id}: chamber_size {stated!r} is not read; the chamber "
                      f"layout is {layout} ({wells} well(s) per chamber)",
                      path=f"dfms.{dfm_id}.params.chamber_size")

        problem = _opto.setting_problem(dfm.get(_opto.SETTING_KEY),
                                        where=f"dfms.{dfm_id}.{_opto.SETTING_KEY}")
        if problem:
            _emit(issues, "error", problem, path=f"dfms.{dfm_id}.{_opto.SETTING_KEY}")
        _lint_program_present(issues, g, dfm_id, dfm, program, data_folder)

        chambers = dfm.get("chambers") or dfm.get("Chambers")
        ## Per-DFM constraints the Experiment Type imposes (Progressive
        ## Ratio's Chamber Groups: paired_chambers, or chamber_groups, or
        ## what the Opto Program's Linkage makes, and same-treatment pairs),
        ## after what the Opto Program fills in, exactly as the loader does it.
        try:
            from . import experiment_types as _et
            _type = _et.get_experiment_type(g.get("experiment_type"), path.parent)
        except Exception:  # noqa: BLE001 — the type problem is reported elsewhere
            _type = None
        if _type is not None:
            _assign = ({int(k): str(v) for k, v in chambers.items()}
                       if isinstance(chambers, dict) else {})
            section = program.section(dfm_id) if program is not None else None
            if section is not None:
                from types import SimpleNamespace

                p_all = {**(g_params if isinstance(g_params, dict) else {}),
                         **(d_params if isinstance(d_params, dict) else {})}
                physical = SimpleNamespace(
                    chamber_sets=p_all.get("chamber_sets"),
                    pi_direction=str(p_all.get("pi_direction") or "left").lower())
                dfm, notes = _type.complete_dfm_node(dfm_id, dict(dfm), params=physical,
                                                     dfm_program=section)
                for note in notes:
                    if "disagrees" in note or "well B" in note:
                        _emit(issues, "warning", note, path=f"dfms.{dfm_id}")
            for problem in _type.validate_dfm(dfm_id, dfm, _assign):
                _emit(issues, "error", problem, path=f"dfms.{dfm_id}")
        if chambers is None:
            _emit(issues, "warning", f"DFM {dfm_id}: no 'chambers' assignments",
                  path=f"dfms.{dfm_id}.chambers")
        elif factor_names and isinstance(chambers, dict):
            for ch_idx, raw in chambers.items():
                levels = [s.strip() for s in str(raw).split(",")]
                if len(levels) != len(factor_names):
                    _emit(
                        issues, "error",
                        f"DFM {dfm_id} chamber {ch_idx}: expected {len(factor_names)} factor "
                        f"levels ({', '.join(factor_names)}), got {len(levels)}: {raw!r}",
                        path=f"dfms.{dfm_id}.chambers.{ch_idx}",
                    )
                    continue
                for fname, lvl in zip(factor_names, levels, strict=True):
                    allowed = factor_levels.get(fname) or []
                    if allowed and lvl not in allowed:
                        _emit(
                            issues, "error",
                            f"DFM {dfm_id} chamber {ch_idx}: level {lvl!r} not in "
                            f"declared {fname} levels {allowed}",
                            path=f"dfms.{dfm_id}.chambers.{ch_idx}",
                        )

    return issues


#: Rows of a DFM's first data file read to tell its Generation.  The
#: firmware fills ``Phase`` on every 4.0.0 row, the lead-in included, so the
#: first rows say what the whole file is without loading it.
_GENERATION_SAMPLE_ROWS = 300


def _file_generation(data_dir: Path, dfm_id: int) -> int | None:
    """The Generation of *dfm_id*'s data, from the header and first rows of
    its first ``DFM<id>_*.csv``; ``None`` when there is none or it cannot
    be read (the loader reports that)."""
    import pandas as pd

    from .paradigm_state import detect_generation
    from .utils import natural_sorted

    files = natural_sorted(Path(data_dir).glob(f"DFM{int(dfm_id)}_*.csv"))
    if not files:
        return None
    try:
        head = pd.read_csv(files[0], nrows=_GENERATION_SAMPLE_ROWS)
    except (OSError, ValueError):
        return None
    return detect_generation(head)


def _lint_data_folder(issues: list[LintIssue], path: Path, cfg: dict) -> Path | None:
    """``data_folder:`` (ADR-0021): where the loader will read this config's
    recording, or ``None`` when it would refuse to.

    A location the loader refuses — missing, a file, holding the experiment
    folder, or beside a recording of the experiment's own — is an error, in
    the loader's own words.  An absolute value works but is not portable, and
    a backslash is a separator on Windows only, so both are warnings.  Read
    from this file's own key, as the loader reads the config it is given.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    from .layout import DATA_FOLDER_KEY, data_location

    location = data_location(path.parent, config=cfg)
    value = cfg.get(DATA_FOLDER_KEY)
    if location.referenced and isinstance(value, str):
        ## Said before any error: on Linux a backslashed value names a folder
        ## that does not exist, and this is why.
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            _emit(issues, "warning",
                  f"{DATA_FOLDER_KEY}: {value} is an absolute path.  It works, but it is "
                  f"not portable: it breaks when the share is mounted somewhere else.  "
                  f"Prefer a path relative to this experiment's folder.",
                  path=DATA_FOLDER_KEY)
        if "\\" in value:
            _emit(issues, "warning",
                  f"{DATA_FOLDER_KEY}: {value} contains backslashes.  Write forward "
                  f"slashes — a backslash is a folder separator on Windows only.",
                  path=DATA_FOLDER_KEY)
    if location.problem:
        _emit(issues, "error", location.problem, path=DATA_FOLDER_KEY)
        return None
    return Path(location.path)


def _lint_program_present(issues: list[LintIssue], g: dict, dfm_id: int,
                          dfm: dict, program, data_folder: Path | None) -> None:
    """Decision 4: a firmware 4.0.0 recording (generation 3) names each
    Paradigm row by row, but only ``Program.txt`` gives its type and
    settings, so a DFM without its section is warned of loudly.  Not for a
    DFM whose light QC is switched off (``optogenetics: no``), which reads
    no Paradigm.  The recording is read in *data_folder*, where the loader
    reads it; nothing is said when that location cannot be read."""
    if data_folder is None:
        return
    from . import opto_program as _opto
    from .opto_light import lacking_program, no_program_warning

    setting = dfm.get(_opto.SETTING_KEY)
    if setting is None:
        setting = g.get(_opto.SETTING_KEY)
    try:
        if _opto.normalize_setting(setting) == _opto.SETTING_NO:
            return
    except ValueError:
        pass                                   # reported as an error above
    generation = _file_generation(data_folder, dfm_id)
    if generation is not None and lacking_program({dfm_id: generation}, program):
        _emit(issues, "warning", no_program_warning([dfm_id]), path=f"dfms.{dfm_id}")


def _resolved_layout(issues: list[LintIssue], path: Path, g: dict) -> str | None:
    """The Chamber Layout the loader will use (ADR-0007): the Experiment
    Type's own, else ``chamber_layout:``, else two-well, read from this
    config's ``global:`` over, for a Member, the Project design's, as the
    loader merges them.  ``None`` when it cannot be resolved: an unknown
    type or an unknown ``chamber_layout``, which the loader refuses and
    which are reported here."""
    from . import experiment_types as _et
    from .project import design_for_member

    design_global, _source = design_for_member(path.parent)
    merged = {**(design_global or {}), **g}
    try:
        etype = _et.get_experiment_type(merged.get("experiment_type"), path.parent)
    except ValueError as err:
        _emit(issues, "error", str(err), path="global.experiment_type")
        return None
    try:
        return etype.resolve_chamber_layout(merged)
    except ValueError as err:
        _emit(issues, "error", str(err), path="global.chamber_layout")
        return None


def _lint_omit(issues: list[LintIssue], path: Path, g: dict) -> None:
    """The Selection (``global.omit:``, ADR-0018): this config's own, and,
    for a Member, the Project design's, which its runs read.  A misspelt id
    leaves nothing out, so it is warned of."""
    from .project import PROJECT_FILENAME, read_design
    from .selections import validate_omit

    for problem in validate_omit(g):
        _emit(issues, "warning", problem, path="global.omit")
    project_dir = path.parent.parent
    if (project_dir / PROJECT_FILENAME).is_file():
        design_global = read_design(project_dir).get("global") or {}
        for problem in validate_omit(design_global if isinstance(design_global, dict) else {}):
            _emit(issues, "warning", f"{PROJECT_FILENAME} design: {problem}",
                  path="design.global.omit")


def _lint_slices(issues: list[LintIssue], path: Path, g: dict) -> None:
    """The Slices (``global.slices:``, ADR-0020): this config's own, and, for
    a Member, the Project design's, which its runs read.  A Slice that cannot
    be used is an error: loading it fails, and the Project leaves it out."""
    from . import slices as _slices
    from .experiment_types import get_experiment_type
    from .project import PROJECT_FILENAME, read_design

    keys = ("slices", "also_exclude", "slice_tolerance_min")

    def check(source: dict, where: str, type_dir: Path, prefix: str = "") -> None:
        ## A lab type is read from its copy beside the file that names it:
        ## the design's beside project.yaml, not beside the Member.
        try:
            etype = get_experiment_type(source.get("experiment_type"), type_dir)
        except ValueError:
            etype = None
        for problem in _slices.slice_problems(source, etype):
            _emit(issues, "error", f"{prefix}{problem}", path=where)

    if any(g.get(key) is not None for key in keys):
        check(g, "global.slices", path.parent)
    project_dir = path.parent.parent
    if (project_dir / PROJECT_FILENAME).is_file():
        design_global = read_design(project_dir).get("global") or {}
        if isinstance(design_global, dict) and any(
                design_global.get(key) is not None for key in keys):
            check(design_global, "design.global.slices", project_dir,
                  f"{PROJECT_FILENAME} design: ")


def _lint_facet_source(issues: list[LintIssue], path: Path, g: dict, dfm_items,
                       program, program_path, data_folder: Path | None) -> None:
    """``facet_source`` (decision 7): a known value, not stated under a type
    that owns its Facets, not beside ``facet_cutoffs``, and for ``program``
    and ``program_cycles`` a ``Program.txt`` section for every DFM the loader
    will load — the checks the loader makes, reported all at once."""
    from . import program_facets as pf

    where = f"global.{pf.FACET_SOURCE_KEY}"
    value = g.get(pf.FACET_SOURCE_KEY)
    if value is None:
        return
    try:
        from . import experiment_types as _et
        etype = _et.get_experiment_type(g.get("experiment_type"), path.parent)
    except Exception:  # noqa: BLE001 — the type problem is reported elsewhere
        etype = None
    if etype is not None and etype.owns_facet_source:
        _emit(issues, "error", f"'{pf.FACET_SOURCE_KEY}' is owned by experiment_type "
                               f"'{etype.name}' and must not appear in the config "
                               f"(remove it; its Facets are its own)", path=where)
        return
    problem = pf.source_problem(value, where=where)
    if problem:
        _emit(issues, "error", problem, path=where)
        return
    source = pf.normalize_source(value)
    if not pf.is_program_source(source):
        return
    if g.get("facet_cutoffs") is not None:
        _emit(issues, "error", f"'facet_cutoffs' and '{pf.FACET_SOURCE_KEY}: {source}' "
                               f"cannot both be stated (remove one)", path=where)
    if g.get("facet_labels"):
        _emit(issues, "warning", f"facet_labels is ignored under {pf.FACET_SOURCE_KEY}: "
                                 f"{source}; Program Facets are named by their Paradigm",
              path="global.facet_labels")
    ## The DFMs the loader will load: configured, not the Environmental
    ## Monitor, and with data when the data folder can be read.
    from .project import dfm_ids_in_folder

    ids: list[int] = []
    for raw_id, node in dfm_items:
        try:
            dfm_id = int(raw_id if raw_id is not None else node.get("id"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not is_environmental_monitor(dfm_id):
            ids.append(dfm_id)
    present = set(dfm_ids_in_folder(data_folder)) if data_folder is not None else set()
    if present:
        ids = [d for d in ids if d in present]
    error = None
    if program is None and program_path is not None:
        error = f"{program_path.name} could not be read"
    for problem in pf.program_problems(source, program, ids, program_error=error):
        _emit(issues, "error", problem, path=where)


def main_cli() -> None:
    """Command-line entry: ``pyflic lint <dir | flic_config.yaml>``.

    Runs the schema linter and, for a directory, the migration checks — the
    two together are the whole crossing from the pre-overhaul layout, since
    there is deliberately no migration tool.
    """
    import sys
    if len(sys.argv) != 2:
        print("usage: pyflic-lint <dir | flic_config.yaml>", file=sys.stderr)
        raise SystemExit(2)
    target = Path(sys.argv[1])

    n_err = n_warn = 0
    if target.is_file():
        cfg = target
        issues = lint_flic_config(cfg)
        for issue in issues:
            print(issue.format(cfg))
        n_err += sum(1 for i in issues if i.severity == "error")
        n_warn += sum(1 for i in issues if i.severity == "warning")
    else:
        from .migration_lint import check_tree
        from .project import is_experiment_dir

        for cfg in sorted(target.rglob("flic_config.yaml")):
            issues = lint_flic_config(cfg)
            for issue in issues:
                print(issue.format(cfg))
            n_err += sum(1 for i in issues if i.severity == "error")
            n_warn += sum(1 for i in issues if i.severity == "warning")
        if not is_experiment_dir(target) and \
                not any(target.rglob("flic_config.yaml")):
            print(f"note: no flic_config.yaml found under {target}")

        migration = check_tree(target)
        if migration:
            print("\n--- migration (ADR-0005..0008) ---")
            for issue in migration:
                print(issue.format())
            n_err += sum(1 for i in migration if i.severity == "error")
            n_warn += sum(1 for i in migration if i.severity == "warning")

    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    raise SystemExit(1 if n_err else 0)

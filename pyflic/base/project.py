"""Project: a marker-file parent of member Experiment Directories (ADR-0005).

A directory with a ``project.yaml`` is a Project; its immediate subdirectories
containing a ``flic_config.yaml`` are its **Members**.  The Project owns the
Combined Analysis (stacked *filtered* per-member summaries with an
``Experiment`` column, pooled tests beside a mixed model), the project-level
``plot_specs.yaml`` / ``figures/``, and the Project Report.

Two things here differ deliberately from PyTrackingAnalysis:

* The ``design:`` section is the authority for **every** key under a Member's
  ``global:``, not just the type and the design factors.  ``well_names`` and
  ``transform_licks`` divergence produces a pooled figure that is *wrong* rather
  than merely noisy, and no shorter list had a defensible boundary.
* A Member normally **omits** ``global:`` and inherits it, so that authority
  costs no duplication.  A ``global:`` that is present is validated key by key.

The Design may declare **Slices** (ADR-0020): separate analyses of part of
every Member's recording.  :meth:`Project.for_slice` gives the same Project
seen through one Slice — its Members' outputs read from ``slices/<name>/``,
its own written to ``<project>/slices/<name>/`` — and a Slice's Combined
Analysis pools only the Members whose recording covers it.
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path

import pandas as pd
import yaml

from . import experiment_types, slices, windowing
from .utils import is_environmental_monitor

#: The Paired-Yoked Difference columns the pooled statistics test, between
#: treatments and against zero, when a Member's table has them.
DIFF_TEST_METRICS: tuple[str, ...] = (
    "dLicksA", "dLicksB", "dEventsA", "dEventsB", "dPI", "dMedDurationA", "dPersistA",
)

PROJECT_FILENAME = "project.yaml"
#: The per-Experiment config.  A Project never has one of its own: the shared
#: design lives in ``project.yaml`` and each Member carries this file.
CONFIG_FILENAME = "flic_config.yaml"

#: Keys of ``global:`` the Design owns absolutely (ADR-0005).  Listed rather
#: than inferred so a new global key is a deliberate decision, not a silent
#: extension of the authority.
DESIGN_KEYS: tuple[str, ...] = (
    "experiment_type",
    "chamber_layout",
    "params",
    "well_names",
    "constants",
    "transform_licks",
    "experimental_design_factors",
    "facet_cutoffs",
    "facet_labels",
    "facet_source",
    "exclusion_group",
    ## Slices (ADR-0020): the separate analyses of part of every Member's
    ## recording, the groups the whole recording also excludes, and how far
    ## short of a Slice's end a recording may stop and still cover it.
    "also_exclude",
    "slices",
    "slice_tolerance_min",
    "optogenetics",
    ## The Selection (ADR-0018): what every Member's Run QC, Run analysis and
    ## Generate plots leave out, the same for all of them, since a Project's
    ## Members are analysed alike.
    "omit",
    ## The Treatment control's level, when the Experiment Type has one
    ## (ADR-0019).
    "control_level",
)


def is_project_dir(path) -> bool:
    return os.path.isfile(os.path.join(str(path), PROJECT_FILENAME))


def is_experiment_dir(path) -> bool:
    return os.path.isfile(os.path.join(str(path), CONFIG_FILENAME))


def has_experiment_data(path) -> bool:
    """True when *path* holds a recording: a ``data/`` subdirectory with at
    least one ``DFM*.csv``.  This is the experiment-shape test for listing
    scaffolding candidates; any other subdirectory is ignored."""
    data = os.path.join(str(path), "data")
    try:
        entries = os.listdir(data)
    except OSError:
        return False
    ## The Environmental Monitor's DFM99_<n>.csv alone is not a recording.
    return any(e.upper().startswith("DFM") and e.lower().endswith(".csv")
               and not _is_monitor_name(e) for e in entries)


def _is_monitor_name(name: str) -> bool:
    match = re.match(r"^DFM_?(\d+)", name, flags=re.IGNORECASE)
    return bool(match) and is_environmental_monitor(match.group(1))


def dfm_ids_in_data(path) -> list[int]:
    """DFM ids present in ``path/data``, parsed from ``DFM<id>_<n>.csv``.

    pyflic can discover this where PyTrackingAnalysis cannot, which is what
    lets :meth:`Project.scaffold_member` reconcile a copied ``dfms:`` block
    against the recording that actually exists.
    """
    data = os.path.join(str(path), "data")
    ids: set[int] = set()
    try:
        entries = os.listdir(data)
    except OSError:
        return []
    for entry in entries:
        match = re.match(r"^DFM(\d+)_\d+\.csv$", entry, flags=re.IGNORECASE)
        ## DFM99_<n>.csv is the Environmental Monitor: never scaffolded.
        if match and not is_environmental_monitor(match.group(1)):
            ids.add(int(match.group(1)))
    return sorted(ids)


def _normalize(value):
    """Comparison form for design values: numbers as floats, mappings and
    sequences normalized element-wise, everything else as-is.

    Without this ``20`` and ``20.0`` in two hand-edited configs would read as a
    design violation, and the error would be indefensible.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value


def create_project_file(project_dir, name: str | None = None,
                        notes: str = "", design: dict | None = None) -> str:
    """Write (or update) ``project.yaml``.

    An existing file's unknown keys are preserved; *design* (when given)
    replaces the ``design:`` section.  A file with no ``scripts:`` key at all is
    seeded with the default Project Script named ``batch`` so every Project
    ships a visible, editable default run — matching PyTrackingAnalysis.  An
    existing block is never touched: an empty list is a deliberate deletion and
    re-seeding it would undo the user's edit.
    """
    from .script_editor.project_actions import default_project_script

    path = os.path.join(str(project_dir), PROJECT_FILENAME)
    payload: dict = {}
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    payload["name"] = name or payload.get("name") \
        or os.path.basename(os.path.normpath(project_dir))
    if notes:
        payload["notes"] = notes
    elif "notes" in payload and not notes:
        payload.pop("notes")
    if design is not None:
        payload["design"] = design
    if "scripts" not in payload:
        payload["scripts"] = [default_project_script()]
    os.makedirs(str(project_dir), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return path


def read_design(project_dir) -> dict:
    """The ``design:`` section of *project_dir*'s ``project.yaml``, or ``{}``.

    Deliberately file-level rather than a :class:`Project` method: a Project
    whose Members contradict the Design refuses to construct, and the Design is
    exactly what the caller needs in order to *repair* that.  The config editor
    also reads it for a single Member, with no Project loaded at all.
    """
    marker = os.path.join(str(project_dir), PROJECT_FILENAME)
    if not os.path.isfile(marker):
        return {}
    try:
        with open(marker, encoding="utf-8") as handle:
            meta = yaml.safe_load(handle) or {}
    except Exception:  # noqa: BLE001 - an unreadable project.yaml is not a design
        return {}
    return dict(meta.get("design") or {})


def design_for_member(experiment_dir) -> tuple[dict, str] | tuple[None, None]:
    """The Design governing the Experiment Directory at *experiment_dir*.

    Returns ``(design_global, project_yaml_path)`` when the directory is a
    Member of a Project that declares a Design, and ``(None, None)`` for a
    standalone experiment — which is governed by nothing and stays free.
    """
    parent = os.path.dirname(os.path.abspath(str(experiment_dir)))
    design = read_design(parent)
    design_global = dict(design.get("global") or {})
    if not design_global:
        return None, None
    return design_global, os.path.join(parent, PROJECT_FILENAME)


def members_stating_global(project_dir) -> list[tuple[str, bool]]:
    """Members that carry a ``global:`` block of their own.

    Each entry is ``(member name, agrees with the design)``.  A Member that
    agrees is merely duplicating the Design; one that does not will refuse to
    load, and both are fixed the same way — by deleting the block so the
    Member inherits (:func:`adopt_design`).
    """
    design_global = dict(read_design(project_dir).get("global") or {})
    out: list[tuple[str, bool]] = []
    for entry in sorted(os.listdir(str(project_dir))):
        sub = os.path.join(str(project_dir), entry)
        if not os.path.isdir(sub) or not is_experiment_dir(sub):
            continue
        try:
            with open(os.path.join(sub, CONFIG_FILENAME),
                      encoding="utf-8") as handle:
                cfg = yaml.safe_load(handle) or {}
        except Exception:  # noqa: BLE001 - an unreadable member is not one of these
            continue
        own = cfg.get("global")
        if not own:
            continue
        agrees = all(
            key in DESIGN_KEYS
            and _normalize(value) == _normalize(design_global.get(key))
            for key, value in dict(own).items())
        out.append((entry, agrees))
    return out


def adopt_design(project_dir, names=None) -> list[str]:
    """Delete the ``global:`` block from each named Member so it inherits.

    The Design is the authority (ADR-0005); a Member's own ``global:`` can only
    duplicate it or contradict it.  Removing the block is therefore the repair
    for both, and it is the only edit made — ``dfms:``, ``scripts:`` and every
    other key are written back untouched.

    Returns the names actually changed.
    """
    if not (read_design(project_dir).get("global") or {}):
        ## Without a Design there is nothing to inherit: stripping the blocks
        ## would leave the Members with no global: at all.
        raise ValueError(
            f"{project_dir} declares no design: — write one before asking its "
            f"members to inherit it")
    changed: list[str] = []
    targets = list(names) if names is not None else \
        [name for name, _agrees in members_stating_global(project_dir)]
    for name in targets:
        path = os.path.join(str(project_dir), name, CONFIG_FILENAME)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        if "global" not in cfg:
            continue
        cfg.pop("global")
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
        changed.append(name)
    return changed


#: The light QC tables a Member writes, stacked into ``<project>_LightQC.csv``
#: with a ``Source`` column: the Progressive Ratio light QC (one row per
#: Chamber Group) and the optogenetic light QC every type has (one row per
#: Linkage Group, ``qc/opto/``).
LIGHT_QC_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("PR", ("analysis", "pr_light_qc.csv")),
    ("Opto", ("qc", "opto", "opto_light_qc.csv")),
)
#: The switch that decides whether a Source's failed groups leave.
_EXCLUDE_SWITCH = {"PR": "exclude_failed_pr_groups", "Opto": "exclude_failed_opto_chambers"}

#: The light tables an optogenetic Member writes into ``analysis/``
#: (:mod:`pyflic.base.light_summary`), stacked into ``<project>_<key>.csv``,
#: and with them its yoked comparison
#: (:func:`~pyflic.base.analytics.yoked_comparison`, a yoked design only).
LIGHT_TABLES: dict[str, str] = {
    "LightSummary": "light_summary.csv",
    "LightSummary_Facet": "light_summary_facet.csv",
    "LightWells": "light_wells.csv",
    "LightWells_Facet": "light_wells_facet.csv",
    "YokedComparison": "yoked_comparison.csv",
}


def flagged_light_qc_groups(light_qc: pd.DataFrame) -> pd.DataFrame:
    """The groups a stacked light QC table (``<project>_LightQC.csv``: an
    ``Experiment`` and a ``Source`` column over each Member's tables) flagged,
    one row each.

    ``Source`` is ``PR`` (a Progressive Ratio Chamber Group) or ``Opto`` (an
    optogenetic Linkage Group, ``Wells`` naming its wells); a table without the
    column is all ``PR``.  ``Status`` says what the pooled numbers did with the
    group: *excluded* (its Member's auto-removal took its chambers out),
    *retained* (it failed, but the Source's exclusion switch was off, so it IS
    in the pooled numbers) or *kept — warning only*.  ``LickFreeRunStartMin``
    is minutes since the PR group's training end; ``UnexplainedOnsetMin`` the
    recording minute an Opto group's unexplained light began.
    """
    columns = ["Experiment", "Source", "DFM", "Group", "Wells", "Treatment", "Verdict",
               "Status", "Flags", "LickFreeRunStartMin", "UnexplainedOnsetMin"]
    frame = light_qc.copy()
    if "Source" not in frame.columns:
        frame["Source"] = "PR"
    for column in ("Wells", "LickFreeRunStartMin", "UnexplainedOnsetMin"):
        if column not in frame.columns:
            frame[column] = pd.NA if column == "Wells" else float("nan")
    flags = frame["Flags"].fillna("").astype(str)
    flagged = frame[flags.str.strip() != ""].copy()
    if flagged.empty:
        return pd.DataFrame(columns=columns)
    excluded = flagged["Excluded"].astype(str).str.lower().isin(("true", "1"))
    failed = flagged["Verdict"].astype(str) == "failed"
    flagged["Source"] = flagged["Source"].fillna("PR").astype(str)
    flagged["Status"] = "kept — warning only"
    retained = [f"retained ({_EXCLUDE_SWITCH.get(src, 'exclusion')}: false)"
                for src in flagged["Source"]]
    flagged.loc[failed, "Status"] = [r for r, f in zip(retained, failed) if f]
    flagged.loc[excluded, "Status"] = "excluded"
    flagged["Treatment"] = flagged["Treatment"].fillna("")
    flagged["Wells"] = flagged["Wells"].fillna("")
    return flagged[columns].reset_index(drop=True)


class _NameShim:
    """Duck-typed ``arena`` so an AI payload builder that reads
    ``experiment.arena.experiment_name`` works on a Project unchanged."""

    def __init__(self, name: str):
        self.experiment_name = name


class Project:
    """The loaded Project: member discovery, design validation, the Combined
    Analysis, and the Project Report."""

    def __init__(self, project_dir: str | Path):
        self.project_directory = os.path.abspath(str(project_dir))
        marker = os.path.join(self.project_directory, PROJECT_FILENAME)
        if not os.path.isfile(marker):
            raise FileNotFoundError(
                f"Not a Project: no {PROJECT_FILENAME} in "
                f"{self.project_directory}")
        with open(marker, encoding="utf-8") as handle:
            meta = yaml.safe_load(handle) or {}
        self.meta = meta
        self.name = str(meta.get("name") or os.path.basename(
            self.project_directory))
        self.notes = str(meta.get("notes") or "")
        self.design: dict = dict(meta.get("design") or {})
        self.design_global: dict = dict(self.design.get("global") or {})

        self.analysis_path = os.path.join(self.project_directory, "analysis")
        self.figures_path = os.path.join(self.project_directory, "figures")
        self.arena = _NameShim(self.name)  # AI-payload compatibility
        #: The Slice this view of the Project is (:meth:`for_slice`), or
        #: ``None`` for the whole recording.
        self.slice = None
        self._whole = None

        # ---- discover members -------------------------------------
        self.member_names: list[str] = []
        self.configs: dict[str, dict] = {}
        #: Real paths already taken.  Two symlinked directories pointing at
        #: one recording used to be counted twice — the same chambers analyzed
        #: twice and stacked into the Combined Analysis under two labels
        #: (ADR-0009).  ``layout.members_in`` enforces the same rule.
        seen: set[str] = set()
        for entry in sorted(os.listdir(self.project_directory)):
            sub = os.path.join(self.project_directory, entry)
            if os.path.isdir(sub) and is_experiment_dir(sub):
                real = os.path.realpath(sub)
                if real in seen:
                    continue
                seen.add(real)
                self.member_names.append(entry)
                with open(os.path.join(sub, CONFIG_FILENAME),
                          encoding="utf-8") as handle:
                    self.configs[entry] = yaml.safe_load(handle) or {}
        if not self.member_names and not self.design:
            raise ValueError(
                f"Project '{self.name}' has no members: no subdirectory of "
                f"{self.project_directory} contains a {CONFIG_FILENAME}, and "
                f"there is no design: section to scaffold from")

        ## Lenient here: a malformed script block must not block loading the
        ## Project (the Script Editor and runner surface the specifics).
        def _script_list(key):
            raw = meta.get(key) or []
            return [dict(item) for item in raw
                    if isinstance(item, dict) and item.get("name")]
        self.scripts: list[dict] = _script_list("scripts")
        self.experiment_scripts: list[dict] = _script_list("experiment_scripts")

        self.warnings: list[str] = []
        self._resolve_type()
        self._validate_design_match()

    # ------------------------------------------------------------------
    # Type / design resolution
    # ------------------------------------------------------------------

    def _resolve_type(self) -> None:
        if self.design_global:
            source = self.design_global
        elif self.member_names:
            source = self._own_global(self.member_names[0])
        else:
            source = {}
        ## A lab type is read from the copy beside project.yaml (ADR-0019).
        self.experiment_type = experiment_types.get_experiment_type(
            source.get("experiment_type"), self.project_directory)
        self.chamber_layout = self.experiment_type.resolve_chamber_layout(source)
        self.design_factors = {
            str(k): [str(v) for v in (vals or [])]
            for k, vals in (source.get("experimental_design_factors")
                            or {}).items()
        }
        ## A blank exclusion_group: is general, as the loader reads it; None
        ## here would reach load_member as "exclude nothing".
        self.exclusion_group = slices.whole_recording_group(source)
        ## Slices (ADR-0020): the usable ones; the rest are warnings.
        self.also_exclude = slices.also_exclude(source)
        self.slice_tolerance = slices.tolerance(source)
        self.slices = slices.usable_slices(source, self.experiment_type)
        self.slice_problems = slices.slice_problems(source, self.experiment_type)
        ## Program Facets (facet_source) and data-derived ones differ in their
        ## windows from DFM to DFM, so the pooled rows group by Facet label.
        self.facet_source = self.experiment_type.resolve_facet_source(source)
        self.facets_vary_by_dfm = self.experiment_type.facets_vary_by_dfm(source)

    def _own_global(self, name: str) -> dict:
        """A Member's *own* ``global:`` block as written — not the resolved
        one.  Design validation compares what the file states, so a Member
        that states nothing is conformant by construction."""
        return dict((self.configs[name].get("global") or {}))

    def resolved_global(self, name: str) -> dict:
        """The ``global:`` a Member loads with: the Design's, with its own
        (already validated) keys layered on top."""
        merged = dict(self.design_global)
        merged.update(self._own_global(name))
        return merged

    def _validate_design_match(self) -> None:
        """Hard-fail when a Member contradicts the Design (ADR-0005).

        With a ``design:`` section every key a Member states under
        ``global:`` must equal the Design's.  Without one (a Project assembled
        from standalone experiments), Members must agree with each other —
        the same rule with the first Member standing in for the Design.
        """
        if self.design_global:
            problems = self._validate_against_design()
        else:
            problems = self._validate_agreement()
        if problems:
            raise ValueError(
                f"Project '{self.name}': members do not match the project "
                f"design ({len(problems)} problem(s)):\n  - "
                + "\n  - ".join(problems))
        self._collect_warnings()

    def _validate_against_design(self) -> list[str]:
        problems: list[str] = []
        for name in self.member_names:
            problems += self._design_problems(name, self.configs[name],
                                              self.design_global)
        return problems

    def _design_problems(self, label: str, config: dict,
                         design_global: dict) -> list[str]:
        """Why *config* would not be a conforming Member under *design_global*.

        One config at a time, by design: the same test serves the Members on
        disk at load and a config that is not in the Project yet — one about to
        be copied in from elsewhere — so the second can be refused *before* it
        is written rather than after it has made the Project unloadable.
        """
        from .yaml_config import PHYSICAL_DFM_KEYS, _normalize_param_overrides

        problems: list[str] = []
        for key, value in dict(config.get("global") or {}).items():
            if key not in DESIGN_KEYS:
                ## An unknown global key is not silently blessed: the Design
                ## owns global:, so anything it does not name has nowhere
                ## legitimate to come from.
                problems.append(
                    f"{label}: global key '{key}' is not part of the "
                    f"project design (the design owns global:)")
                continue
            expected = design_global.get(key)
            if key == "experiment_type":
                ## The registry is case-insensitive and the editor writes the
                ## canonical spelling, so a hand-written 'hedonic' under a
                ## design saying 'Hedonic' names the same type.  Comparing the
                ## strings made that a fatal deviation.
                where = getattr(self, "project_directory", None)
                try:
                    same = (experiment_types.get_experiment_type(value, where).name
                            == experiment_types.get_experiment_type(expected, where).name)
                except ValueError:
                    same = False
                if same:
                    continue
            if _normalize(value) != _normalize(expected):
                problems.append(
                    f"{label}: global.{key} is {value!r} but the project "
                    f"design requires {expected!r}")
        ## Per-DFM overrides are checked here too, so a Project fails to load
        ## rather than failing later inside one member's load.
        for node in (config.get("dfms") or []):
            if not isinstance(node, dict):
                continue
            over = node.get("params") or node.get("parameters") or {}
            illegal = sorted(set(_normalize_param_overrides(over))
                             - PHYSICAL_DFM_KEYS)
            if illegal:
                problems.append(
                    f"{label}: DFM {node.get('id')} overrides params "
                    f"{illegal}; only {sorted(PHYSICAL_DFM_KEYS)} may vary "
                    f"per DFM inside a Project")
        return problems

    def design_problems_for(self, config: dict,
                            label: str = "this config") -> list[str]:
        """Why *config* would not be a conforming Member of this Project.

        Empty list means it would be accepted.  Used before a config is copied
        into a Member directory: a non-conforming Member stops the whole
        Project loading, and undoing the copy afterwards is hand work.
        """
        if not isinstance(config, dict):
            return [f"{label}: not a YAML mapping — this is not a "
                    f"{CONFIG_FILENAME}."]
        if "dfms" not in config and "DFMs" not in config:
            return [f"{label}: no 'dfms:' section — this is not a "
                    f"{CONFIG_FILENAME}."]
        if self.design_global:
            return self._design_problems(label, config, self.design_global)
        ## Legacy Project (no design: section): the Members are the authority,
        ## so a newcomer has to agree with the first of them.
        if not self.member_names:
            return []
        reference = self._own_global(self.member_names[0])
        own = dict(config.get("global") or {})
        problems: list[str] = []
        for key in DESIGN_KEYS:
            if key not in reference and key not in own:
                continue
            if _normalize(own.get(key)) != _normalize(reference.get(key)):
                problems.append(
                    f"{label}: global.{key} is {own.get(key)!r} but "
                    f"'{self.member_names[0]}' has {reference.get(key)!r}")
        return problems

    def _validate_agreement(self) -> list[str]:
        """Legacy mode (no design section): Members must agree with the
        first one on every design key they state."""
        problems: list[str] = []
        if not self.member_names:
            return problems
        first = self.member_names[0]
        reference = self._own_global(first)
        for name in self.member_names[1:]:
            own = self._own_global(name)
            for key in DESIGN_KEYS:
                if key not in reference and key not in own:
                    continue
                if _normalize(own.get(key)) != _normalize(reference.get(key)):
                    problems.append(
                        f"{name}: global.{key} is {own.get(key)!r} but "
                        f"'{first}' has {reference.get(key)!r} (add a design: "
                        f"section to project.yaml to make this explicit)")
        return problems

    def _collect_warnings(self) -> None:
        """Non-fatal spread worth surfacing — things the Design leaves free."""
        def spread(getter, label):
            values: dict[str, list[str]] = {}
            for name in self.member_names:
                values.setdefault(repr(getter(name)), []).append(name)
            if len(values) > 1:
                detail = "; ".join(f"{v}: {', '.join(ns)}"
                                   for v, ns in values.items())
                self.warnings.append(
                    f"{label} differs across members ({detail})")

        if len(self.member_names) > 1:
            spread(lambda n: len(self.configs[n].get("dfms") or []),
                   "DFM count")
        for name in self.member_names:
            if not has_experiment_data(self.member_dir(name)):
                self.warnings.append(f"{name}: no DFM CSVs in data/")
        for problem in self.slice_problems:
            self.warnings.append(f"design: {problem}")

    # ------------------------------------------------------------------
    # Slices (ADR-0020)
    # ------------------------------------------------------------------

    @property
    def slice_name(self) -> str | None:
        return None if self.slice is None else self.slice.name

    @property
    def whole(self) -> "Project":
        """The Project seen through the whole recording."""
        return self if self._whole is None else self._whole

    def for_slice(self, name: str | None) -> "Project":
        """This Project seen through the Slice *name* (``None``: the whole
        recording).  Members load as that Slice, their saved outputs are read
        from ``slices/<name>/``, and the Combined Analysis, the Project Report
        and the figures are written to ``<project>/slices/<name>/``."""
        whole = self.whole
        if name is None:
            return whole
        chosen = next((s for s in whole.slices if s.name == name), None)
        if chosen is None:
            raise slices.SliceError(f"Project '{whole.name}': the design declares no usable "
                                    f"Slice named '{name}'")
        view = copy.copy(whole)
        view._whole = whole
        view.slice = chosen
        root = slices.output_dir(whole.project_directory, chosen.name)
        view.analysis_path = str(root / "analysis")
        view.figures_path = str(root / "figures")
        view.exclusion_group = chosen.exclusion_group
        view.also_exclude = chosen.also_exclude
        return view

    def slice_views(self) -> list["Project"]:
        """The whole recording, then every usable Slice."""
        whole = self.whole
        return [whole] + [whole.for_slice(s.name) for s in whole.slices]

    @property
    def transform_licks(self) -> bool:
        """Whether the Members' Licks are on the fourth-root scale: the
        Design's ``transform_licks``, true when unset, as the loader reads it."""
        return bool(self.design_global.get("transform_licks", True))

    @property
    def exclusion_groups(self) -> tuple[str, ...]:
        """The groups this view excludes: its own first, then also_exclude."""
        return slices.exclusion_groups(self.exclusion_group, self.also_exclude)

    def view_label(self) -> str:
        """``"whole recording"`` or ``"Slice early (0, 360] min"``."""
        if self.slice is None:
            return "whole recording"
        return f"Slice {self.slice.name} {self.slice.bounds_label()}"

    def member_output_dir(self, name: str) -> str:
        """Where a Member's outputs for this view live: its directory, or
        ``slices/<name>/`` inside it."""
        if self.slice is None:
            return self.member_dir(name)
        return str(slices.output_dir(self.member_dir(name), self.slice.name))

    def slice_record(self, name: str) -> dict | None:
        """The ``slice.json`` the Member's last run of this Slice left."""
        if self.slice is None:
            return None
        return slices.read_record(self.member_dir(name), self.slice.name)

    def _current_record(self, name: str) -> dict | None:
        """:meth:`slice_record`, or ``None`` when it was left by a run of
        another definition of the Slice: whether the Member covers the Slice
        as the Design now states it is then unknown until it runs again."""
        record = self.slice_record(name)
        if record is None or slices.definition_changed(record, self.slice):
            return None
        return record

    def not_covered(self) -> dict[str, str]:
        """``{member: reason}`` for the Members whose recording does not
        cover this Slice, as their last run of it recorded."""
        out: dict[str, str] = {}
        if self.slice is None:
            return out
        for name in self.member_names:
            record = self._current_record(name)
            if record is not None and not record.get("covered", True):
                out[name] = str(record.get("reason") or "does not cover the Slice")
        return out

    def slice_coverage_frame(self) -> pd.DataFrame | None:
        """One row per Member for this Slice: covered or not, where its
        data ends, and why it was left out; ``None`` for the whole recording."""
        if self.slice is None:
            return None
        rows = []
        for name in self.member_names:
            record = self._current_record(name)
            if record is None:
                why = ("this Slice has not been run for the Member"
                       if self.slice_record(name) is None else
                       "the Slice has changed since the Member last ran it")
                rows.append({"Experiment": name, "Covered": "not run", "ActualEndMin": None,
                             "Reason": why})
                continue
            rows.append({"Experiment": name,
                         "Covered": "yes" if record.get("covered", True) else "no",
                         "ActualEndMin": record.get("actual_end_minutes"),
                         "Reason": record.get("reason") or ""})
        return pd.DataFrame(rows, columns=["Experiment", "Covered", "ActualEndMin", "Reason"])

    def open_end_spread_note(self) -> str | None:
        """For an open-ended Slice, a warning when the pooled Members'
        recordings end more than the tolerance apart: the Slice then pools
        unequal spans under one name."""
        if self.slice is None or not self.slice.open_end:
            return None
        ends = {}
        for name in self.member_names:
            record = self.slice_record(name)
            if record and record.get("covered", True) and \
                    record.get("actual_end_minutes") is not None:
                ends[name] = float(record["actual_end_minutes"])
        if len(ends) < 2:
            return None
        low, high = min(ends.values()), max(ends.values())
        if high - low <= self.slice_tolerance:
            return None
        detail = ", ".join(f"{n} {v:.1f}" for n, v in sorted(ends.items()))
        return (f"Slice '{self.slice.name}' runs to the end of each recording, and the pooled "
                f"Members end {high - low:.1f} min apart (more than the {self.slice_tolerance:g}-min "
                f"tolerance): {detail}")

    # ------------------------------------------------------------------
    # Scripts
    # ------------------------------------------------------------------

    def find_script(self, name: str) -> dict | None:
        for script in self.scripts:
            if script.get("name") == name:
                return script
        return None

    def find_experiment_script(self, name: str) -> dict | None:
        """The centrally-held Experiment Script *name*, or None (the
        ``run_in_experiments`` bridge then falls back to each Member's own)."""
        for script in self.experiment_scripts:
            if script.get("name") == name:
                return script
        return None

    # ------------------------------------------------------------------
    # Member access / scaffolding
    # ------------------------------------------------------------------

    def member_dir(self, name: str) -> str:
        return os.path.join(self.project_directory, name)

    def load_member(self, name: str, **kwargs):
        """Load a Member with the Design's ``global:`` supplied for
        inheritance and per-DFM overrides restricted (ADR-0005).  A Slice view
        loads the Member as its Slice unless ``slice=`` says otherwise."""
        from .yaml_config import load_experiment_yaml

        if self.slice is not None:
            kwargs.setdefault("slice", self.slice.name)
        kwargs.setdefault("exclusion_group", self.whole.exclusion_group)
        exp = load_experiment_yaml(
            self.member_dir(name),
            design_global=self.design_global or None,
            in_project=True,
            **kwargs,
        )
        ## project.yaml is the authority: a Slice the Member loads next reads
        ## the design as it is then, a Selection tick since included.
        directory = self.project_directory
        exp.design_source = lambda: dict(read_design(directory).get("global") or {})
        return exp

    #: Pre-ADR-0009 spellings.  ``Project.experiment_names`` and friends are
    #: the names the sibling app still uses, and notebooks written against
    #: pyflic before the Member rename call them; keeping them costs three
    #: lines and saves every one of those from breaking.
    @property
    def experiment_names(self) -> list[str]:
        return self.member_names

    def experiment_dir(self, name: str) -> str:
        return self.member_dir(name)

    def load_experiment(self, name: str, **kwargs):
        return self.load_member(name, **kwargs)

    def experiment_status(self, name: str) -> dict:
        return self.member_status(name)

    # ------------------------------------------------------------------
    # Layout: what a run can use, and what is blocked (ADR-0009)
    # ------------------------------------------------------------------

    def member_layouts(self) -> list:
        """Every experiment-shaped subdirectory, classified — healthy or
        Blocked (:mod:`pyflic.base.layout`).

        The Project's own membership test asks only "is there a
        ``flic_config.yaml``".  This asks the harder question a *run* asks —
        "is there data the loader can find" — so an Unfiled Recording and a
        configured folder with no data are visible in the Project panel
        instead of failing at load, an hour into an unattended run.
        """
        from . import layout

        return layout.members_in(self.project_directory)

    def blocked_members(self) -> list:
        """Members a run cannot use as they stand."""
        return [item for item in self.member_layouts() if item.blocked]

    def unconfigured_dirs(self) -> list[str]:
        """Immediate subdirectories that hold a recording but are not
        Members yet — a ``data/`` with at least one ``DFM*.csv`` and no
        ``flic_config.yaml``.  These are the scaffolding candidates.

        Deliberately *not* the Unfiled Recordings (ADR-0009): scaffolding
        reconciles the copied ``dfms:`` block against the DFM ids actually in
        ``data/``, and doing that before the files are filed would write a
        config reconciled against nothing.  File first, then scaffold —
        :meth:`unfiled_members` names the ones waiting on that.
        """
        from . import layout

        return [item.name for item in self.member_layouts()
                if item.status == layout.NO_CONFIG]

    def unfiled_members(self) -> list[str]:
        """Subdirectories whose DFM CSVs sit at their root, not in ``data/``.

        One click fixes them (:func:`pyflic.base.layout.file_recording`), and
        until it does they are invisible to the Project: the loader reads
        ``data/`` and nothing else.
        """
        from . import layout

        return [item.name for item in self.member_layouts()
                if item.status == layout.UNFILED]

    def scaffold_member_config(self, name: str) -> tuple[dict, list[str]]:
        """A ``flic_config.yaml`` for Member *name*, plus reconciliation notes.

        The ``dfms:`` block is copied from the first existing Member —
        members of one design almost always reuse the plate layout, and
        retyping 48 chamber assignments is the work worth avoiding — then
        reconciled against the DFM ids actually present in ``data/``:

        * an id in the data with no entry is **added**, chambers unassigned;
        * an entry with no data is **flagged**, never silently dropped, because
          a missing CSV is usually a copy that has not finished.

        ``global:`` is deliberately absent: the Member inherits the Design.
        """
        import copy

        notes: list[str] = []
        template: list = []
        if self.member_names:
            source = self.member_names[0]
            template = copy.deepcopy(self.configs[source].get("dfms") or [])
            notes.append(f"dfms: copied from '{source}'")
        found = dfm_ids_in_data(self.member_dir(name))
        if not found:
            notes.append("data/: no DFM CSVs found — nothing to reconcile")
            return ({"dfms": template} if template else {"dfms": []}), notes

        by_id: dict[int, dict] = {}
        for node in template:
            if isinstance(node, dict) and node.get("id") is not None:
                by_id[int(node["id"])] = node
        for missing in sorted(set(by_id) - set(found)):
            notes.append(
                f"DFM {missing}: in the copied layout but absent from data/ — "
                f"kept and flagged, not removed")
        blank_chambers = self._blank_chamber_block(template)
        for extra in sorted(set(found) - set(by_id)):
            by_id[extra] = {"id": extra, "chambers": dict(blank_chambers)}
            notes.append(
                f"DFM {extra}: present in data/ but not in the copied layout — "
                f"added with chambers unassigned")
        dfms = [by_id[i] for i in sorted(by_id)]
        return {"dfms": dfms}, notes

    def _blank_chamber_block(self, template: list) -> dict:
        """An unassigned chamber mapping shaped like the template's.

        With no template — the first Member of a Project that so far is only a
        Design — the shape comes from the Design's Chamber Layout, because a
        single-well experiment has twelve chambers and a two-well six.  A fixed
        six here gave a single-well member half a plate.
        """
        for node in template:
            if isinstance(node, dict) and isinstance(node.get("chambers"), dict):
                return {k: "" for k in node["chambers"]}
        n_chambers = 12 if self.chamber_layout == "single_well" else 6
        return {i: "" for i in range(1, n_chambers + 1)}

    def scaffold_member(self, name: str) -> tuple[str, list[str]]:
        """Give Member *name* a design-conformant ``flic_config.yaml``.

        Creates the directory and its ``data/`` when missing, so this serves
        both "add a new member" and "adopt a folder already sitting in the
        Project".  Never overwrites an existing config — a member's chamber
        assignments are hand-made.
        """
        if not name or name != os.path.basename(name) or name in (".", ".."):
            ## The name is joined onto the Project directory, so a separator
            ## in it writes a config (and a data/ folder) outside the Project.
            raise ValueError(
                f"'{name}' is not a member name — it must be a single folder "
                f"name inside the Project")
        directory = self.member_dir(name)
        config_path = os.path.join(directory, CONFIG_FILENAME)
        if os.path.isfile(config_path):
            raise FileExistsError(f"'{name}' already has a {CONFIG_FILENAME}")
        os.makedirs(os.path.join(directory, "data"), exist_ok=True)
        config, notes = self.scaffold_member_config(name)
        with open(config_path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
        self.member_names.append(name)
        self.member_names.sort()
        self.configs[name] = config
        return config_path, notes

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def member_report_path(self, name: str) -> str:
        """Where a Member's own experiment report is written
        (``pdf_report.write_experiment_report``'s default)."""
        return os.path.join(self.member_output_dir(name), "analysis", "experiment_report.pdf")

    def member_status(self, name: str) -> dict:
        """Cheap per-Member status from saved artifacts (no data load).

        In a Slice view, ``covered`` is what the Member's last run of the
        Slice recorded (``None``: never run) and ``reason`` why it was left
        out; ``stale`` also holds when the Design's Slice no longer matches
        the one that run used."""
        directory = self.member_dir(name)
        analysis = os.path.join(self.member_output_dir(name), "analysis")
        summary = os.path.join(analysis, "feeding_summary.csv")
        status = {
            "analyzed": os.path.isfile(summary),
            "faceted": os.path.isfile(
                os.path.join(analysis, "feeding_summary_facet.csv")),
            "report": os.path.isfile(self.member_report_path(name)),
            "has_data": has_experiment_data(directory),
            "dfms": len(dfm_ids_in_data(directory)),
            "chambers": None,
            ## The stale rule (ADR-0010): saved results computed BEFORE the
            ## current exclusion declaration describe a chamber population
            ## nobody asked for.  Showing them a date beside "analyzed" is
            ## worse than showing nothing — the date is true and the number
            ## it stands for is not.
            "stale": False,
            "covered": True,
            "reason": None,
        }
        if self.slice is not None:
            ## A record of another definition says nothing of this one, not
            ## even whether the recording covers it: what that run wrote is
            ## stale, and coverage is unknown until the Slice runs again.
            record = self.slice_record(name)
            current = self._current_record(name)
            status["covered"] = None if current is None else bool(current.get("covered", True))
            status["reason"] = None if current is None else current.get("reason")
            if status["covered"] is False:
                status["analyzed"] = False
            elif record is not None and current is None:
                status["stale"] = True
        if status["analyzed"]:
            try:
                status["chambers"] = len(pd.read_csv(summary))
            except Exception:  # noqa: BLE001
                pass
            declaration = os.path.join(directory, "remove_chambers.csv")
            try:
                status["stale"] = status["stale"] or (
                    os.path.isfile(declaration)
                    and os.path.getmtime(declaration)
                    > os.path.getmtime(summary))
            except OSError:
                pass
        return status

    def run_all(self, *, make_reports: bool = True,
                skip_analyzed: bool = False, log=print,
                all_slices: bool = True) -> list[str]:
        """Run each Member's basic analysis (and report); returns failures.

        With *all_slices* (the default) a Project with Slices runs the whole
        recording and then every Slice of every Member (ADR-0020), so each
        Slice's Combined Analysis has every Member that covers it.  A Member
        that does not cover a Slice is recorded as such, not failed."""
        failures: list[str] = []
        views = self.slice_views() if all_slices else [self]
        for view in views:
            if view.slice is not None:
                log(f"--- {view.view_label()} ---")
            for name in self.member_names:
                tag = name if view.slice is None else f"{name} / {view.slice.name}"
                if skip_analyzed and view.member_status(name)["analyzed"]:
                    log(f"[{tag}] already analyzed — skipped")
                    continue
                log(f"[{tag}] running analysis...")
                try:
                    exp = view.load_experiment(name)
                except slices.SliceNotCovered as err:
                    slices.record_not_covered(view.member_dir(name), err)
                    log(f"[{tag}] not covered — left out of this Slice: {err.reason}")
                    continue
                except Exception as err:  # noqa: BLE001
                    failures.append(f"{tag}: {err}")
                    log(f"[{tag}] FAILED: {type(err).__name__}: {err}")
                    continue
                try:
                    exp.execute_basic_analysis()
                    if make_reports:
                        from .pdf_report import write_experiment_report
                        write_experiment_report(exp)
                except Exception as err:  # noqa: BLE001
                    failures.append(f"{tag}: {err}")
                    log(f"[{tag}] FAILED: {type(err).__name__}: {err}")
        return failures

    # ------------------------------------------------------------------
    # Combined Analysis
    # ------------------------------------------------------------------

    def combined_frames(self):
        """The stacked, filtered Member summaries.

        Returns ``(summary, facet, missing)``.  Each frame carries an
        ``Experiment`` first column; ``missing`` lists Members with no saved
        summary — they are omitted, never silently analyzed, because pooling
        half a Project without saying so is worse than refusing.
        """
        summaries, facets, missing = [], [], []
        ## A Slice pools only the Members that cover it (ADR-0020): one left
        ## out is named by not_covered(), not counted as missing.
        left_out = self.not_covered()
        for name in self.member_names:
            if name in left_out:
                continue
            path = os.path.join(self.member_output_dir(name), "analysis",
                                "feeding_summary.csv")
            if not os.path.isfile(path):
                missing.append(name)
                continue
            df = pd.read_csv(path)
            df.insert(0, "Experiment", name)
            summaries.append(df)
            fpath = os.path.join(self.member_output_dir(name), "analysis",
                                 "feeding_summary_facet.csv")
            if os.path.isfile(fpath):
                fdf = pd.read_csv(fpath)
                fdf.insert(0, "Experiment", name)
                facets.append(fdf)
        summary = pd.concat(summaries, ignore_index=True) if summaries else None
        facet = pd.concat(facets, ignore_index=True) if facets else None
        return summary, facet, missing

    @staticmethod
    def unfaceted_members(summary: pd.DataFrame | None,
                          facet: pd.DataFrame | None) -> list[str]:
        """The pooled Members with no ``feeding_summary_facet.csv`` beside
        their summary while others have one: they are in the whole-recording
        statistics and missing from every Facet's."""
        if summary is None or facet is None or "Experiment" not in facet.columns:
            return []
        faceted = set(facet["Experiment"].astype(str))
        return [name for name in dict.fromkeys(summary["Experiment"].astype(str))
                if name not in faceted]

    def combined_diff_frame(self):
        """The stacked Members' ``paired_yoked_diff.csv`` (Progressive Ratio
        only), with an ``Experiment`` first column, or ``None`` when no Member
        has one."""
        frames = []
        for name in self._pooled_members():
            path = os.path.join(self.member_output_dir(name), "analysis",
                                "paired_yoked_diff.csv")
            if os.path.isfile(path):
                df = pd.read_csv(path)
                df.insert(0, "Experiment", name)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def combined_light_qc_frame(self):
        """The stacked Members' light QC tables — each Progressive Ratio
        Member's ``analysis/pr_light_qc.csv`` and every optogenetic Member's
        ``qc/opto/opto_light_qc.csv`` (:data:`LIGHT_QC_SOURCES`) — with
        ``Experiment`` and ``Source`` first, or ``None`` when no Member has
        one."""
        frames = []
        for name in self._pooled_members():
            for source, parts in LIGHT_QC_SOURCES:
                path = os.path.join(self.member_output_dir(name), *parts)
                if os.path.isfile(path):
                    df = pd.read_csv(path)
                    df.insert(0, "Source", source)
                    df.insert(0, "Experiment", name)
                    frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def combined_breaking_point_frame(self):
        """The stacked Members' ``pr_breaking_point.csv`` (Progressive Ratio
        only; ADR-0014), with an ``Experiment`` first column, or ``None`` when
        no Member has one — a Member analysed before the table existed has
        none, and is not guessed at."""
        frames = []
        for name in self._pooled_members():
            path = os.path.join(self.member_output_dir(name), "analysis",
                                "pr_breaking_point.csv")
            if os.path.isfile(path):
                df = pd.read_csv(path)
                df.insert(0, "Experiment", name)
                frames.append(df)
        return pd.concat(frames, ignore_index=True) if frames else None

    def combined_light_frames(self) -> dict[str, pd.DataFrame]:
        """The stacked Members' light tables (:data:`LIGHT_TABLES`), each with
        an ``Experiment`` first column, keyed ``LightSummary``,
        ``LightSummary_Facet``, ``LightWells``, ``LightWells_Facet`` and
        ``YokedComparison``; a table no Member wrote is absent."""
        out: dict[str, pd.DataFrame] = {}
        for key, filename in LIGHT_TABLES.items():
            frames = []
            for name in self._pooled_members():
                path = os.path.join(self.member_output_dir(name), "analysis", filename)
                if os.path.isfile(path):
                    df = pd.read_csv(path)
                    df.insert(0, "Experiment", name)
                    frames.append(df)
            if frames:
                out[key] = pd.concat(frames, ignore_index=True)
        return out

    def member_generations(self, name: str) -> list[int]:
        """The Generations of a Member's DFMs, from its saved outputs (no data
        load): the light table's ``Generation`` column, else the ``-
        generation:`` lines of its ``summary.txt``; empty when neither says."""
        analysis = os.path.join(self.member_output_dir(name), "analysis")
        for filename in (LIGHT_TABLES["LightSummary"], LIGHT_TABLES["LightWells"]):
            path = os.path.join(analysis, filename)
            if os.path.isfile(path):
                try:
                    values = pd.to_numeric(pd.read_csv(path, usecols=["Generation"])
                                           ["Generation"], errors="coerce").dropna()
                    if not values.empty:
                        return sorted({int(v) for v in values})
                except (ValueError, KeyError, OSError):
                    pass
        path = os.path.join(analysis, "summary.txt")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    found = re.findall(r"^- generation: (\d+)", handle.read(), re.MULTILINE)
                return sorted({int(v) for v in found})
            except OSError:
                pass
        return []

    def generation_2_members(self) -> list[str]:
        """Members recorded (in part) from DFM firmware older than 4.0.0, so
        with no Training information (decision 15): a generation-2 row in
        their saved light table or ``summary.txt``, or a Progressive Ratio
        Member's saved Facet ``Recording`` (the one Facet such a Member's
        chamber groups have)."""
        out = []
        for name in self.member_names:
            if 2 in self.member_generations(name):
                out.append(name)
                continue
            path = os.path.join(self.member_output_dir(name), "analysis",
                                "feeding_summary_facet.csv")
            if os.path.isfile(path):
                try:
                    labels = set(pd.read_csv(path, usecols=["Facet"])["Facet"].astype(str))
                except (ValueError, KeyError, OSError):
                    labels = set()
                if "Recording" in labels:
                    out.append(name)
        return out

    def flagged_groups(self, light_qc: pd.DataFrame | None = None) -> pd.DataFrame | None:
        """Every Chamber Group the light QC flagged, across Members
        (:func:`flagged_light_qc_groups`), or ``None`` when no Member has a
        light QC table."""
        if light_qc is None:
            light_qc = self.combined_light_qc_frame()
        return None if light_qc is None else flagged_light_qc_groups(light_qc)

    def aggregated_exclusions(self) -> pd.DataFrame:
        """Every Member's exclusions in one table (ADR-0005).

        Sources: each Member's ``remove_chambers.csv`` rows for the groups
        this view excludes (its own and ``also_exclude``), and the
        auto-removal table saved by a run.  A Slice reads its own.
        """
        from .exclusions import read_exclusions

        rows: list[dict] = []
        for name in self._pooled_members():
            directory = self.member_dir(name)
            groups = read_exclusions(directory)
            seen: set[tuple[int, int]] = set()
            for group in self.exclusion_groups:
                for dfm_id, chambers in (groups.get(group) or {}).items():
                    for chamber in chambers:
                        if (int(dfm_id), int(chamber)) in seen:
                            continue
                        seen.add((int(dfm_id), int(chamber)))
                        rows.append({"Experiment": name, "DFM": int(dfm_id),
                                     "Chamber": int(chamber), "Source": "manual",
                                     "Note": f"group '{group}'"})
            auto = os.path.join(self.member_output_dir(name), "analysis",
                                "removed_chambers.csv")
            if os.path.isfile(auto):
                try:
                    adf = pd.read_csv(auto)
                    for _, row in adf.iterrows():
                        rows.append({
                            "Experiment": name,
                            "DFM": int(row.get("DFM", 0)),
                            "Chamber": int(row.get("Chamber", 0)),
                            "Source": "auto",
                            "Note": str(row.get("Reason", "") or ""),
                        })
                except Exception:  # noqa: BLE001
                    pass
        columns = ["Experiment", "DFM", "Chamber", "Source", "Note"]
        return pd.DataFrame(rows, columns=columns)

    def _pooled_members(self) -> list[str]:
        """The Members this view pools: all of them, less any whose
        recording does not cover the Slice."""
        left_out = self.not_covered()
        return [n for n in self.member_names if n not in left_out]

    def build_combined_analysis(self) -> dict:
        """Write the Combined Analysis into ``<project>/analysis/`` (a Slice
        view: ``<project>/slices/<name>/analysis/``)."""
        summary, facet, missing = self.combined_frames()
        left_out = self.not_covered()
        if summary is None:
            where = "" if self.slice is None else f" for {self.view_label()}"
            raise ValueError(
                f"No member has a saved analysis{where} yet (missing: "
                f"{', '.join(missing) or 'none'}"
                + (f"; not covered: {', '.join(left_out)}" if left_out else "")
                + "). Run the experiments first.")
        os.makedirs(self.analysis_path, exist_ok=True)
        written: list[str] = []
        coverage = self.slice_coverage_frame()
        if coverage is not None:
            path = os.path.join(self.analysis_path, f"{self.name}_SliceCoverage.csv")
            coverage.to_csv(path, index=False, na_rep="NA")
            written.append(path)

        path = os.path.join(self.analysis_path, f"{self.name}_Summary.csv")
        summary.to_csv(path, index=False, na_rep="NA")
        written.append(path)
        if facet is not None:
            path = os.path.join(self.analysis_path,
                                f"{self.name}_Summary_Facet.csv")
            facet.to_csv(path, index=False, na_rep="NA")
            written.append(path)
        diff = self.combined_diff_frame()
        if diff is not None:
            path = os.path.join(self.analysis_path,
                                f"{self.name}_PairedYokedDiff.csv")
            diff.to_csv(path, index=False, na_rep="NA")
            written.append(path)
        light_qc = self.combined_light_qc_frame()
        if light_qc is not None:
            path = os.path.join(self.analysis_path, f"{self.name}_LightQC.csv")
            light_qc.to_csv(path, index=False, na_rep="NA")
            written.append(path)
        breaking = self.combined_breaking_point_frame()
        if breaking is not None:
            path = os.path.join(self.analysis_path, f"{self.name}_BreakingPoint.csv")
            breaking.to_csv(path, index=False, na_rep="NA")
            written.append(path)
        light = self.combined_light_frames()
        for key in LIGHT_TABLES:
            path = os.path.join(self.analysis_path, f"{self.name}_{key}.csv")
            if key in light:
                light[key].to_csv(path, index=False, na_rep="NA")
                written.append(path)
            elif os.path.isfile(path):
                ## The light figures draw from whichever of these exists, so
                ## one no Member backs any more must not outlive the rebuild.
                os.remove(path)
        path = os.path.join(self.analysis_path, f"{self.name}_Excluded.csv")
        self.aggregated_exclusions().to_csv(path, index=False, na_rep="NA")
        written.append(path)
        path = os.path.join(self.analysis_path, f"{self.name}_Stats.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(self.stats_text(summary, facet, diff))
        written.append(path)
        ## An AI narrative is a derivative of one Combined Analysis; a stale
        ## one must not sit beside fresh numbers.
        from .ai import delete_project_narrative

        delete_project_narrative(self)
        return {"written": written, "missing": missing, "not_covered": left_out}

    def build_all_combined_analyses(self, log=print) -> dict[str | None, dict]:
        """The Combined Analysis of the whole recording and of every Slice
        (ADR-0020), keyed by Slice name (``None``: the whole recording).  A
        Slice no Member has been run for is skipped and logged; the whole
        recording raises as :meth:`build_combined_analysis` does."""
        results: dict[str | None, dict] = {}
        for view in self.slice_views():
            if view.slice is None:
                results[None] = view.build_combined_analysis()
                continue
            try:
                results[view.slice.name] = view.build_combined_analysis()
            except ValueError as err:
                log(f"[{view.view_label()}] skipped: {err}")
        return results

    # ------------------------------------------------------------------
    # Facets shared across Members
    # ------------------------------------------------------------------

    def shared_windows(self):
        """``(windows, labels)`` for the Design's facets, or ``(None, None)``.

        Because the Design owns ``facet_cutoffs``, every Member is windowed
        identically by construction — there is nothing to reconcile, which is
        the point of ADR-0008.
        """
        cutoffs = self.experiment_type.resolve_facet_cutoffs(self.design_global)
        if not cutoffs:
            return None, None
        windows = list(windowing.facet_windows(cutoffs))
        labels = self.experiment_type.phase_labels_for(windows,
                                                       self.design_global)
        return windows, labels

    def window_labels(self, facet: pd.DataFrame) -> tuple[list, dict]:
        """Ordered windows present in *facet* and their display labels."""
        windows: list[tuple] = []
        for raw in facet["FacetRange"]:
            w = windowing.parse_range(raw)
            if w not in windows:
                windows.append(w)
        shared, labels = self.shared_windows()
        if shared is not None and self.slice is not None:
            ## A Slice's Members cut each Facet to the Slice and keep its name
            ## (ADR-0020), so the design's windows are cut the same way here.
            kept = [(slices.intersect_window(w, self.slice.bounds), label)
                    for w, label in zip(shared, labels)]
            shared = [w for w, _label in kept if w is not None]
            labels = [label for w, label in kept if w is not None]
        if shared is not None and [tuple(w) for w in shared] == windows:
            return windows, dict(zip(windows, labels))
        return windows, {w: windowing.minute_label(w) for w in windows}

    # ------------------------------------------------------------------
    # Statistics: pooled per-chamber tests + nested mixed model
    # ------------------------------------------------------------------

    def metrics(self) -> list[str]:
        """The metrics the pooled statistics report, by Chamber Layout."""
        if self.chamber_layout == "two_well":
            return ["PI", "EventPI", "Licks", "Events", "MedDuration"]
        return ["Licks", "Events", "MedDuration", "MeanDuration"]

    def comparison_rows(self, summary: pd.DataFrame,
                        facet: pd.DataFrame | None) -> list[dict]:
        """One row per metric x facet x treatment pair: the pooled per-chamber
        p-value beside the nested mixed-model p-value."""
        frames = self._facet_frames(summary, facet)
        return self._compare_frames(frames, self.metrics(), transformed=self.transform_licks)

    def _facet_frames(self, summary: pd.DataFrame,
                      facet: pd.DataFrame | None) -> list[tuple[str, pd.DataFrame]]:
        """``[(phase label, rows), ...]`` — one per Facet, else the whole
        recording.  A type whose windows come from the data (ADR-0013), and
        Program Facets (``facet_source``), are split on the ``Facet`` label,
        because their ``FacetRange`` differs per Chamber Group or per DFM and
        would fragment one phase into many; a Program Facet pools every
        Member's chambers that ran the same Paradigm."""
        frames: list[tuple[str, pd.DataFrame]] = []
        if facet is None or facet.empty:
            return [(slices.whole_label(self.slice), summary)]
        if self.facets_vary_by_dfm and "Facet" in facet.columns:
            order = (list(self.experiment_type.phase_labels)
                     if getattr(self.experiment_type, "data_derived_facets", False) else [])
            labels = list(dict.fromkeys(facet["Facet"].astype(str)))
            labels.sort(key=lambda s: order.index(s) if s in order else len(order))
            for label in labels:
                frames.append((label, facet[facet["Facet"].astype(str) == label]))
            return frames
        if "FacetRange" in facet.columns:
            windows, label_of = self.window_labels(facet)
            for w in windows:
                mask = facet["FacetRange"].map(windowing.parse_range) == w
                frames.append((label_of[w], facet[mask]))
            return frames
        return [(slices.whole_label(self.slice), summary)]

    def diff_comparison_rows(self, diff: pd.DataFrame | None) -> list[dict]:
        """Treatment comparisons on the pooled Paired-Yoked Difference table,
        one observation per Chamber Group, for the type's report Facets."""
        if diff is None or diff.empty:
            return []
        facets = self.experiment_type.report_facets() or \
            list(dict.fromkeys(diff["Facet"].astype(str)))
        frames = [(label, diff[diff["Facet"].astype(str) == label])
                  for label in facets]
        metrics = [c for c in DIFF_TEST_METRICS if c in diff.columns]
        return self._compare_frames(frames, metrics)

    def diff_zero_rows(self, diff: pd.DataFrame | None) -> list[dict]:
        """Per treatment, is the pooled Paired-Yoked Difference non-zero?  One
        observation per Chamber Group, for the type's report Facets
        (:func:`pyflic.base.analytics.zero_tests`): the paired t-test and the
        signed-rank test, with the mixed model's intercept when more than one
        Member is pooled."""
        if diff is None or diff.empty:
            return []
        from .analytics import zero_tests

        facets = self.experiment_type.report_facets() or \
            list(dict.fromkeys(diff["Facet"].astype(str)))
        frames = [(label, diff[diff["Facet"].astype(str) == label])
                  for label in facets]
        metrics = [c for c in DIFF_TEST_METRICS if c in diff.columns]
        return zero_tests(frames, metrics, mixed_p0=self._mixed_p0)

    def breaking_point_rows(self, breaking: pd.DataFrame | None) -> list[dict]:
        """Treatment comparisons of the pooled Breaking Point, one observation
        per Chamber Group: the pooled test and the mixed model as for any
        metric, plus the log-rank test that honours censoring
        (:func:`pyflic.base.analytics.breaking_point_comparisons`)."""
        if breaking is None or breaking.empty:
            return []
        from .analytics import breaking_point_comparisons

        return breaking_point_comparisons(breaking, mixed_p=self._mixed_p)

    def _compare_frames(self, frames: list[tuple[str, pd.DataFrame]],
                        metrics: list[str], *, transformed: bool = False) -> list[dict]:
        from .analytics import treatment_comparisons

        return treatment_comparisons(frames, metrics, mixed_p=self._mixed_p,
                                     transformed=transformed)

    @staticmethod
    def _mixed_p(data: pd.DataFrame, a: str, b: str) -> float | None:
        """Treatment p-value from a linear mixed model on the ``(a, b)`` subset.

        **DFM is nested within Experiment** (ADR-0005): ``groups=Experiment``
        with a DFM variance component.  DFM ids repeat across Members — DFM 1
        of one recording is a different physical device from DFM 1 of another —
        so grouping the stacked frame on DFM alone would merge them into one
        random-effect group and understate the treatment standard error.

        Returns ``None`` rather than raising when the fit fails, so one
        non-converging metric never costs the whole stats table.
        """
        try:
            import warnings

            import statsmodels.formula.api as smf

            sub = data[data["Treatment"].isin([a, b])].copy()
            if sub["Experiment"].nunique() < 2:
                return None
            sub["is_b"] = (sub["Treatment"] == b).astype(float)
            sub["DFM"] = sub["DFM"].astype(str)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = smf.mixedlm(
                        "Value ~ is_b", sub, groups=sub["Experiment"],
                        vc_formula={"DFM": "0 + C(DFM)"},
                    ).fit(reml=True, method="lbfgs")
                except Exception:  # noqa: BLE001
                    ## Two variance components can fail to converge on a small
                    ## Project. Falling back to the Experiment level alone is
                    ## still better than reporting the pooled p twice.
                    fit = smf.mixedlm(
                        "Value ~ is_b", sub, groups=sub["Experiment"],
                    ).fit(reml=True, method="lbfgs")
            return float(fit.pvalues["is_b"])
        except Exception:  # noqa: BLE001
            return None

    @staticmethod
    def _mixed_p0(data: pd.DataFrame) -> float | None:
        """Intercept p-value of a linear mixed model on one treatment's
        Paired-Yoked Differences: is the mean difference non-zero once
        between-member and between-device variation is allowed for?  DFM
        nested within Experiment, as in :meth:`_mixed_p`; ``None`` when the
        fit fails or only one Member is present."""
        try:
            import warnings

            import statsmodels.formula.api as smf

            sub = data.copy()
            if sub["Experiment"].nunique() < 2:
                return None
            sub["DFM"] = sub["DFM"].astype(str)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = smf.mixedlm(
                        "Value ~ 1", sub, groups=sub["Experiment"],
                        vc_formula={"DFM": "0 + C(DFM)"},
                    ).fit(reml=True, method="lbfgs")
                except Exception:  # noqa: BLE001
                    fit = smf.mixedlm(
                        "Value ~ 1", sub, groups=sub["Experiment"],
                    ).fit(reml=True, method="lbfgs")
            p = float(fit.pvalues["Intercept"])
            return p if p == p else None
        except Exception:  # noqa: BLE001
            return None

    def _generation_2_lines(self) -> list[str]:
        from .light_summary import NO_TRAINING_INFORMATION

        return [f"  {name} [generation 2]: {NO_TRAINING_INFORMATION}"
                for name in self.generation_2_members()]

    def _flagged_groups_lines(self) -> list[str]:
        """The Stats text's "Flagged Chamber Groups" block — the Progressive
        Ratio light QC and the optogenetic light QC, and every Member without
        Training information (generation 2, decision 15) — or nothing when no
        Member has a light QC table or a generation-2 recording."""
        flagged = self.flagged_groups()
        generation_2 = self._generation_2_lines()
        if flagged is None and not generation_2:
            return []
        out = ["Flagged Chamber Groups (light QC: progressive ratio and optogenetic)"]
        if flagged is None:
            return out + generation_2 + [""]
        if flagged.empty:
            out += ["  (none — every group's light was earned by its licks)",
                    *generation_2, ""]
            return out
        for row in flagged.itertuples(index=False):
            if row.Source == "Opto":
                where = (f"{row.Experiment} DFM {int(row.DFM)} linkage group "
                         f"{row.Group}" + (f" [{row.Wells}]" if row.Wells else ""))
            else:
                where = f"{row.Experiment} DFM {int(row.DFM)} group {int(row.Group)}"
            if row.Treatment:
                where += f" ({row.Treatment})"
            onset = ""
            if pd.notna(row.LickFreeRunStartMin):
                onset = (f"; lick-free from {float(row.LickFreeRunStartMin):.1f} min "
                         f"after training end")
            elif pd.notna(row.UnexplainedOnsetMin):
                onset = f"; unexplained from minute {float(row.UnexplainedOnsetMin):.0f}"
            out.append(f"  {where} [{row.Source}]: {row.Status} — {row.Flags}{onset}")
        retained = int((flagged["Status"].str.startswith("retained")).sum())
        if retained:
            out.append(f"  {retained} failed group(s) retained by member setting ARE "
                       f"in the pooled numbers below.")
        out.extend(generation_2)
        out.append("")
        return out

    def light_headline_metrics(self, light: pd.DataFrame | None = None) -> list[str]:
        """The light table's headline metrics for this Project (decision 9
        of ``docs/paradigm-analyses-plan.md``), pooled per Paradigm Family:
        those of the families every Member's ``data/Program.txt`` runs on
        the DFMs its saved light table records as generation 3
        (:func:`~pyflic.base.light_summary.program_headline_metrics`).
        :data:`~pyflic.base.light_summary.HEADLINE_METRICS` when no Member
        has such a DFM.  *light* is the stacked ``LightSummary``, read when
        not given."""
        from . import opto_program as op
        from .light_summary import program_headline_metrics
        from .paradigm_state import GENERATION_PARADIGM

        if light is None:
            light = self.combined_light_frames().get("LightSummary")
        programs = []
        if light is not None and {"Experiment", "DFM", "Generation"} <= set(light.columns):
            generation = pd.to_numeric(light["Generation"], errors="coerce")
            current = light[generation == GENERATION_PARADIGM]
            for name, rows in current.groupby("Experiment", sort=False):
                try:
                    path = op.find_program(os.path.join(self.member_dir(str(name)), "data"))
                    program = op.read_program(path) if path is not None else None
                except (op.ProgramError, OSError, ValueError):
                    program = None
                ids = sorted({int(d) for d in pd.to_numeric(rows["DFM"], errors="coerce").dropna()})
                programs.append((program, ids))
        return program_headline_metrics(programs)

    def light_comparison_rows(self, light: pd.DataFrame | None = None,
                              light_facet: pd.DataFrame | None = None) -> list[dict]:
        """Treatment comparisons of the light table's headline metrics
        (:meth:`light_headline_metrics`), one observation per chamber, per
        Facet as the feeding metrics are (pooled test beside the mixed
        model).  A Progressive Ratio table keeps its paired chambers
        (:func:`~pyflic.base.light_summary.behaviour_rows`).  Reads the
        stacked tables when not given; empty without a light table."""
        from .light_summary import behaviour_rows

        if light is None and light_facet is None:
            frames = self.combined_light_frames()
            light = frames.get("LightSummary")
            light_facet = frames.get("LightSummary_Facet")
        light = behaviour_rows(light)
        if light is None or light.empty:
            return []
        frames = self._facet_frames(light, behaviour_rows(light_facet))
        return self._compare_frames(frames, self.light_headline_metrics(light))

    def stats_text(self, summary: pd.DataFrame,
                   facet: pd.DataFrame | None,
                   diff: pd.DataFrame | None = None) -> str:
        rows = self.comparison_rows(summary, facet)
        if diff is None and self.experiment_type.has_module("ratio"):
            diff = self.combined_diff_frame()
        diff_rows = self.diff_comparison_rows(diff)
        bar = "=" * 72
        n_exp = (summary["Experiment"].nunique()
                 if "Experiment" in summary.columns else 1)
        out = [bar, f"Combined Analysis — {self.name}", bar, ""]
        if self.slice is not None:
            out.append(f"Slice             : {self.slice.name} {self.slice.bounds_label()}, "
                       f"each Member analysed as if its recording were only these minutes")
        out.append(f"Members pooled : {n_exp} "
                   f"({', '.join(self._pooled_members())})")
        for name, reason in self.not_covered().items():
            out.append(f"Not covered       : {name}, left out of this Slice ({reason})")
        for name in self.unfaceted_members(summary, facet):
            out.append(f"Not faceted       : {name} has no feeding_summary_facet.csv, so it "
                       f"is missing from every Facet's statistics (re-run its analysis if it "
                       f"should have one)")
        if note := self.open_end_spread_note():
            out.append(f"WARNING           : {note}")
        out.append(f"Chambers pooled   : {len(summary)}")
        out.append(f"Experiment type   : {self.experiment_type.display_name}")
        out.append(f"Chamber layout    : {self.chamber_layout}")
        from . import program_facets

        if program_facets.is_program_source(self.facet_source):
            out.append(f"Facets            : {program_facets.SOURCE_LABELS[self.facet_source]}"
                       f" — pooled by Paradigm label across members")
        out.append("")
        out.append("p_pooled : per-chamber test across all members "
                   "(Welch for 2 groups, Tukey HSD otherwise) — matches the "
                   "pooled figures.")
        out.append("p_mixed  : linear mixed model, treatment fixed, DFM nested "
                   "within Experiment — accounts for between-member and "
                   "between-device variation.")
        out.append("")
        out.extend(self._flagged_groups_lines())

        def _table(table_rows: list[dict], *, logrank: bool = False) -> list[str]:
            ## A Program Facet is named by its Paradigm, which can be long (a
            ## generation-1 program's generated names), and the light table's
            ## metric names are longer than the feeding ones (AcclimationEndMin):
            ## both columns widen to fit rather than running into the next.
            fw = max([16] + [len(str(row["phase"])) + 1 for row in table_rows])
            w = max([14, *(len(str(r["metric"])) + 1 for r in table_rows)])
            header = (f"{'Metric':<{w}}{'Facet':<{fw}}{'A':<12}{'B':<12}"
                      f"{'nA':>4}{'nB':>5}{'diff':>10}{'p_pooled':>11}"
                      f"{'p_mixed':>10}" + (f"{'p_logrank':>11}" if logrank else ""))
            lines = [header, "-" * len(header)]
            for row in table_rows:
                p_mixed = ("      n/a" if row["p_mixed"] is None
                           else f"{row['p_mixed']:>10.4g}")
                p_logrank = ""
                if logrank:
                    p_logrank = ("        n/a" if row.get("p_logrank") is None
                                 else f"{row['p_logrank']:>11.4g}")
                star = " *" if row["significant"] else ""
                lines.append(
                    f"{row['metric']:<{w}}{row['phase']:<{fw}}{row['a']:<12}"
                    f"{row['b']:<12}{row['n_a']:>4}{row['n_b']:>5}"
                    f"{row['diff']:>10.4g}{row['p_pooled']:>11.4g}{p_mixed}"
                    f"{p_logrank}{star}")
            return lines

        def _zero_table(table_rows: list[dict]) -> list[str]:
            header = (f"{'Metric':<14}{'Facet':<10}{'Treatment':<14}{'n':>4}"
                      f"{'mean':>11}{'sem':>10}{'p_t':>10}{'p_wilcoxon':>12}"
                      f"{'p_mixed':>10}")
            lines = [header, "-" * len(header)]
            for row in table_rows:
                p_w = "n/a" if row["p_wilcoxon"] is None else f"{row['p_wilcoxon']:.4g}"
                p_m = "n/a" if row["p_mixed"] is None else f"{row['p_mixed']:.4g}"
                star = " *" if row["significant"] else ""
                lines.append(
                    f"{row['metric']:<14}{row['phase']:<10}{row['treatment']:<14}"
                    f"{row['n']:>4}{row['mean']:>11.4g}{row['sem']:>10.4g}"
                    f"{row['p_t']:>10.4g}{p_w:>12}{p_m:>10}{star}")
            return lines

        if diff is not None:
            from .experiment_types.progressive_ratio import pooled_yoked_controls

            if diff.empty and not pooled_yoked_controls(self):
                ## Decision 13: a design of solo groups has no difference to
                ## take; said once, where it would have been.
                out.append("Paired − yoked difference: none; the design has no yoked "
                           "controls (every chamber group is one chamber, lit by its own "
                           "fly alone)")
                out.append("")
            else:
                ## Progressive Ratio: the within-group difference is the primary
                ## result (one observation per Chamber Group); per-chamber tables
                ## follow as the secondary section.
                out.append("Paired − yoked difference (primary; one observation "
                           "per chamber group)")
                ## One row per group per Facet: count the groups.
                keys = [c for c in ("Experiment", "DFM", "Group") if c in diff.columns]
                n_groups = len(diff.drop_duplicates(keys)) if keys else len(diff)
                out.append(f"Chamber groups pooled : {n_groups}")
                out.append("")
                if diff_rows:
                    out.extend(_table(diff_rows))
                else:
                    out.append("(no comparable treatment groups found)")
                out.append("")
                out.append("Paired − yoked difference against zero (is the paired fly "
                           "different from its yoked partner?)")
                out.append("p_t: paired t-test, a one-sample t-test on the differences; "
                           "p_wilcoxon: signed-rank test; p_mixed: mixed-model intercept, "
                           "DFM nested within Experiment.")
                out.append("")
                zero_rows = self.diff_zero_rows(diff)
                if zero_rows:
                    out.extend(_zero_table(zero_rows))
                    out.append("* p_t < 0.05")
                else:
                    out.append("(no treatment with two or more differences to test)")
                out.append("")
            breaking = self.combined_breaking_point_frame()
            if breaking is not None:
                from .analytics import as_bool

                censored = (int(as_bool(breaking["Censored"]).sum())
                            if "Censored" in breaking.columns else 0)
                out.append("Breaking point (paired fly; one observation per chamber "
                           "group)")
                out.append(f"Chamber groups pooled : {len(breaking)} "
                           f"({censored} censored)")
                ## Firmware 4.0.0 records the Requirement each light event met
                ## (stream 4): BreakRequirement beside BreakingPoint, medians.
                from .experiment_types.progressive_ratio import break_requirement_view

                req = break_requirement_view(breaking)
                if req is not None:
                    out.append("Break requirement     : " + "; ".join(
                        f"{r['Treatment']} BreakingPoint {r['Breaking point']:g}, "
                        f"BreakRequirement {r['Break requirement']:g} "
                        f"(median, {r['With a requirement']} group(s))"
                        for _, r in req.iterrows()))
                out.append("p_logrank: pairwise log-rank test on the ratio reached, "
                           "treating a censored count as a lower bound; p_pooled and "
                           "p_mixed enter it as observed.")
                out.append("")
                bp_rows = self.breaking_point_rows(breaking)
                if bp_rows:
                    out.extend(_table(bp_rows, logrank=True))
                else:
                    out.append("(no comparable treatment groups found)")
                out.append("")
            out.append("Per-chamber metrics (secondary)")
            out.append("")
        if not rows:
            out.append("(no comparable treatment groups found)")
        else:
            out.extend(_table(rows))
            out.append("")
            out.append("* p_pooled < 0.05")
        out.extend(self._light_lines(_table))
        return "\n".join(out) + "\n"

    def _light_lines(self, table) -> list[str]:
        """The Stats text's "Light behaviour" block: the light table's
        headline metrics compared between treatments, per Facet, or nothing
        when no Member wrote a light table."""
        from .light_summary import behaviour_rows

        frames = self.combined_light_frames()
        light = frames.get("LightSummary")
        if light is None:
            return []
        pooled = behaviour_rows(light)
        out = ["", "Light behaviour (optogenetic members; one observation per chamber)",
               f"Chambers pooled : {len(pooled)}"
               + (" (paired chambers: a yoked fly's light is its partner's)"
                  if len(pooled) < len(light) else "")]
        out.append("")
        rows = self.light_comparison_rows(light, frames.get("LightSummary_Facet"))
        if rows:
            out.extend(table(rows))
            out.append("")
            out.append("* p_pooled < 0.05")
        else:
            out.append("(no comparable treatment groups found)")
        return out

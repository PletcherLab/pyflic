from __future__ import annotations

import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import yaml

from .dfm import DFM, EmptyRangeError
from .experiment import Experiment
from .experiment_design import ExperimentDesign
from .hedonic_experiment import HedonicFeedingExperiment
from .parameters import Parameters
from .progressive_ratio_experiment import ProgressiveRatioExperiment
from .single_well_experiment import SingleWellExperiment
from .treatment import Treatment
from .two_well_experiment import TwoWellExperiment
from .utils import ENVIRONMENTAL_MONITOR_MESSAGE, is_environmental_monitor

#: Chamber Layout -> the ``Experiment`` subclass that computes its metrics
#: (ADR-0007).  This is the *data shape* map; an Experiment Type may name a
#: further subclass via ``experiment_class`` when it brings analysis of its own.
_LAYOUT_CLASS_MAP: dict[str, type[Experiment]] = {
    "single_well": SingleWellExperiment,
    "two_well": TwoWellExperiment,
}

#: Per-DFM ``params:`` overrides that survive inside a Project.  These describe
#: hardware, not analysis: which side of the chamber the reference well sits on
#: and which wells are wired together.  Overriding anything else would
#: reintroduce, one level lower and less visibly, the divergence the Project
#: Design outlaws (ADR-0005).
PHYSICAL_DFM_KEYS: frozenset[str] = frozenset({"pi_direction", "chamber_sets"})


def _resolve_experiment_class(exp_type, chamber_layout: str) -> type[Experiment]:
    """The ``Experiment`` subclass for *exp_type* on *chamber_layout*."""
    dotted = getattr(exp_type, "experiment_class", None)
    if dotted:
        module_name, _, attr = str(dotted).partition(":")
        import importlib

        return getattr(importlib.import_module(module_name), attr)
    return _LAYOUT_CLASS_MAP.get(chamber_layout, Experiment)


def _norm_key(k: str) -> str:
    k = str(k).strip()
    k = k.replace(".", "_").replace("-", "_").replace(" ", "_")
    while "__" in k:
        k = k.replace("__", "_")
    return k.lower()


_PARAM_ALIASES: dict[str, str] = {
    # Baseline
    "baseline_window_minutes": "baseline_window_minutes",
    "baseline_window_min": "baseline_window_minutes",
    "baseline_window": "baseline_window_minutes",
    # Feeding
    "feeding_threshold": "feeding_threshold",
    "feeding_minimum": "feeding_minimum",
    "feeding_minevents": "feeding_minevents",
    "feeding_event_link_gap": "feeding_event_link_gap",
    "link_gap": "feeding_event_link_gap",
    # Tasting
    "tasting_minimum": "tasting_minimum",
    "tasting_maximum": "tasting_maximum",
    "tasting_low": "tasting_minimum",
    "tasting_high": "tasting_maximum",
    "tasting_minevents": "tasting_minevents",
    # Hardware / chamber
    "samples_per_second": "samples_per_second",
    "samples_per_sec": "samples_per_second",
    "chamber_sets": "chamber_sets",
    "chamber_size": "chamber_size",
    "correct_for_dual_feeding": "correct_for_dual_feeding",
    "pi_direction": "pi_direction",
    # Back-compat: accept numeric `pi_multiplier` and interpret 1->left, other->right.
    "pi_multiplier": "pi_direction",
}


def _normalize_param_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if not overrides:
        return {}
    out: dict[str, Any] = {}
    for k, v in overrides.items():
        nk = _norm_key(k)
        nk = _PARAM_ALIASES.get(nk, nk)
        out[nk] = v

    # Coercions
    if "chamber_sets" in out and out["chamber_sets"] is not None:
        out["chamber_sets"] = np.asarray(out["chamber_sets"], dtype=int)
    if "chamber_size" in out and out["chamber_size"] is not None:
        out["chamber_size"] = int(out["chamber_size"])
    if "feeding_minevents" in out and out["feeding_minevents"] is not None:
        out["feeding_minevents"] = int(out["feeding_minevents"])
    if "tasting_minevents" in out and out["tasting_minevents"] is not None:
        out["tasting_minevents"] = int(out["tasting_minevents"])
    if "pi_direction" in out and out["pi_direction"] is not None:
        v = out["pi_direction"]
        # If legacy numeric was supplied under `pi_multiplier`, it arrives here via alias mapping.
        if isinstance(v, (int, float)):
            out["pi_direction"] = "left" if int(v) == 1 else "right"
        else:
            s = str(v).strip().lower()
            if s not in ("left", "right"):
                raise ValueError("pi_direction must be 'left' or 'right'.")
            out["pi_direction"] = s
    if "correct_for_dual_feeding" in out and out["correct_for_dual_feeding"] is not None:
        out["correct_for_dual_feeding"] = bool(out["correct_for_dual_feeding"])
    return out


def _parse_chamber_assignments(chambers_node: Any) -> dict[int, str]:
    """
    Accept either:
    - mapping: {1: "DrugA", 2: "Vehicle"}
    - list: [{index: 1, treatment: "DrugA"}, ...]
    """

    if chambers_node is None:
        return {}
    if isinstance(chambers_node, Mapping):
        return {int(k): str(v) for k, v in chambers_node.items()}
    if isinstance(chambers_node, list):
        out: dict[int, str] = {}
        for item in chambers_node:
            if not isinstance(item, Mapping):
                raise ValueError("Each chambers[] entry must be a mapping with index and treatment.")
            idx = int(item.get("index"))
            trt = str(item.get("treatment"))
            out[idx] = trt
        return out
    raise ValueError("chambers must be a mapping or a list of {index,treatment}.")


def _parse_chamber_factor_assignments(
    chambers_node: Any, factor_names: list[str]
) -> tuple[dict[int, str], dict[int, dict[str, str]]]:
    """
    Parse chamber assignments when ``experimental_design_factors`` is defined.

    Chamber values are comma-separated factor levels in the same order as
    *factor_names*.  The treatment name is the levels joined with ``_``.

    Returns ``(chamber_assignments, chamber_factor_levels)`` where:
      - ``chamber_assignments``: {chamber_idx: treatment_name}
      - ``chamber_factor_levels``: {chamber_idx: {factor_name: level}}
    """

    def _parse_one(idx: int, raw: str) -> tuple[str, dict[str, str]]:
        levels = [s.strip() for s in str(raw).split(",")]
        if len(levels) != len(factor_names):
            raise ValueError(
                f"Chamber {idx}: expected {len(factor_names)} factor level(s) "
                f"({', '.join(factor_names)}), got {len(levels)}: {raw!r}"
            )
        return "_".join(levels), dict(zip(factor_names, levels))

    assignments: dict[int, str] = {}
    factor_levels: dict[int, dict[str, str]] = {}

    if chambers_node is None:
        return assignments, factor_levels
    if isinstance(chambers_node, Mapping):
        for k, v in chambers_node.items():
            idx = int(k)
            trt, fl = _parse_one(idx, v)
            assignments[idx] = trt
            factor_levels[idx] = fl
    elif isinstance(chambers_node, list):
        for item in chambers_node:
            if not isinstance(item, Mapping):
                raise ValueError("Each chambers[] entry must be a mapping.")
            idx = int(item.get("index"))
            raw = item.get("treatment", item.get("levels", ""))
            trt, fl = _parse_one(idx, raw)
            assignments[idx] = trt
            factor_levels[idx] = fl
    else:
        raise ValueError("chambers must be a mapping or a list.")
    return assignments, factor_levels


def _safe_setting(value) -> str:
    """An ``optogenetics:`` value as auto / yes / no, an invalid one as auto
    (the type's validation has already reported it)."""
    from . import opto_program as _opto

    try:
        return _opto.normalize_setting(value)
    except ValueError:
        return _opto.SETTING_AUTO


#: ``exclusion_group``'s default: the whole recording's own group as the
#: design names it (``global.exclusion_group``; ``general`` when unset).
DESIGN_GROUP: Any = object()


def _cache_signature(global_cfg: Mapping[str, Any], excluded: Mapping[int, Sequence[int]],
                     the_slice) -> str:
    """What, beyond the config file and the data, shaped a loaded summary.
    The Selection (``omit:``) and the other Slices' definitions do not, so
    ticking a box or editing another Slice keeps the entry."""
    import json

    return json.dumps({
        "global": {k: v for k, v in global_cfg.items() if k not in ("omit", "slices")},
        "excluded": {str(k): sorted(int(c) for c in v) for k, v in sorted(excluded.items())},
        "slice": None if the_slice is None else the_slice.definition(),
    }, sort_keys=True, default=str)


def _load_dfm_for_config(
    dfm_id: int, params: Parameters, data_dir: str | Path, range_minutes: Sequence[float]
) -> DFM | EmptyRangeError:
    # Separate top-level helper so process pools can pickle it.  A range with
    # no rows of this DFM comes back as the error rather than raised: a Slice
    # load reads it as "this recording does not cover the Slice".
    try:
        return DFM.load(dfm_id, params, data_dir=data_dir, range_minutes=range_minutes)
    except EmptyRangeError as err:
        return err


def load_experiment_yaml(
    experiment_dir: str | Path,
    *,
    config_name: str = "flic_config.yaml",
    range_minutes: Sequence[float] = (0, 0),
    parallel: bool = True,
    max_workers: int | None = None,
    executor: Literal["threads", "processes"] = "threads",
    eager: bool = True,
    use_disk_cache: bool = True,
    exclusion_group: str | None = DESIGN_GROUP,
    design_global: Mapping[str, Any] | None = None,
    in_project: bool = False,
    strict_type: bool = True,
    slice: str | None = None,
) -> Experiment:
    """
    Load an experiment from a experiment directory.

    Reads ``experiment_dir/<config_name>`` (default ``flic_config.yaml``) and
    loads DFM data from ``experiment_dir/data`` — or, when the config names a
    ``data_folder:`` (ADR-0021), from that folder, resolved against
    *experiment_dir* and only ever read.  A ``data_folder:`` the loader cannot
    use raises :class:`~pyflic.base.layout.DataLocationError`.

    Parameters
    ----------
    experiment_dir:
        Project root directory.  Must contain the selected config file.
        Data is read from *experiment_dir/data*, or the config's
        ``data_folder:``.  The returned ``Experiment``
        stores this so that downstream helpers (``write_qc_reports``,
        ``write_summary``, ``_auto_save_fig``) write to
        ``experiment_dir/<config_stem>/qc`` and
        ``experiment_dir/<config_stem>/analysis`` automatically.
    config_name:
        Filename of the YAML config inside *experiment_dir*.  An Experiment
        Directory holds exactly one (ADR-0005); outputs always land in
        ``experiment_dir/analysis``.
    design_global:
        The Project Design's ``global:`` block, supplied when loading a
        Member.  A Member normally omits ``global:`` and inherits this;
        keys it does state win, having already been validated by ``Project``.
    in_project:
        True when loading a Member.  Restricts per-DFM ``params:`` overrides
        to the physical keys (ADR-0005).
    exclusion_group:
        The ``remove_chambers.csv`` group whose chambers are excluded, beside
        the design's ``also_exclude`` groups.  By default the design's own
        ``exclusion_group`` (``general`` when it names none).  ``None``
        excludes nothing, loading the design as written; the experiment
        still names the design's group as the one its removals go to.
    strict_type:
        Raise when the config violates its Experiment Type instead of warning.
        The linter passes False so it can report every problem at once.
    slice:
        The name of a Slice the design declares (ADR-0020): the recording is
        clipped to it at load, its own exclusion group and ``also_exclude``
        groups replace the whole recording's, its ``constants:`` overrides
        are layered on, and outputs go to ``slices/<name>/``.  Raises
        :class:`~pyflic.base.slices.SliceNotCovered` when the recording does
        not reach the Slice's end.  ``None`` (the default) is the whole
        recording.

    Expected YAML structure::

        global:
          params:
            chamber_size: 2
        dfms:
          - id: 1
            params: { feeding_threshold: 20 }
            chambers:
              1: DrugA
              2: Vehicle
          - id: 5
            params: {}
            chambers:
              - index: 1
                treatment: DrugA
    """

    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

    resolved_experiment_dir = Path(experiment_dir).expanduser().resolve()
    path = resolved_experiment_dir / config_name
    if not path.exists():
        raise FileNotFoundError(
            f"{config_name} not found in experiment directory: {resolved_experiment_dir}"
        )
    cfg = yaml.safe_load(path.read_text())
    if not isinstance(cfg, Mapping):
        raise ValueError("YAML root must be a mapping/object.")

    ## Where the recording is (ADR-0021): the experiment's own data/, or the
    ## folder its data_folder: names — the classifier's own answer, so the two
    ## cannot read different folders.  A location the loader cannot use stops
    ## here, before anything is read or written, naming the key, the value as
    ## written and the resolved path.
    from . import layout as _layout

    data_location = _layout.require_data_location(
        resolved_experiment_dir, config=dict(cfg), config_path=str(path))
    data_dir = Path(data_location.path)
    #: How messages name the recording folder: "data/" as always, or, for a
    #: referenced one, where it is and the key as written.
    data_label = data_location.describe() if data_location.referenced else "data/"

    global_cfg = dict(cfg.get("global", {}) or {})
    ## Design inheritance (ADR-0005): inside a Project a Member normally
    ## omits `global:` entirely and inherits the Design's. A `global:` that IS
    ## present has already been validated key-by-key by Project, so merging
    ## here is a no-op for a conformant Member and simply fills the gaps for
    ## one that states only part of the block.
    if design_global:
        merged = dict(design_global)
        merged.update(global_cfg)
        global_cfg = merged

    from . import experiment_types as _experiment_types

    ## A lab type is read from its copy beside the data (ADR-0019).
    exp_type = _experiment_types.get_experiment_type(
        global_cfg.get("experiment_type"), resolved_experiment_dir)
    if getattr(exp_type, "resolved_from", None) == "library":
        print(f"  NOTE: experiment_type '{exp_type.name}' was read from the lab library: "
              f"the data carries no copy of it ({_experiment_types.COPY_NAME}), so "
              f"another machine cannot analyse it the same way.  Save the design or "
              f"the config again to copy it.", flush=True)
    type_problems = exp_type.validate(global_cfg)
    if type_problems and strict_type:
        raise ValueError(
            f"{path.name} does not satisfy experiment_type "
            f"'{exp_type.name}':\n  - " + "\n  - ".join(type_problems))
    for problem in type_problems:
        print(f"  WARNING: {problem}", flush=True)
    chamber_layout = exp_type.resolve_chamber_layout(global_cfg)
    derived_chamber_size = _experiment_types.LAYOUT_CHAMBER_SIZE[chamber_layout]

    global_params_node = global_cfg.get("params", global_cfg.get("parameters", None))
    global_overrides = _normalize_param_overrides(global_params_node)
    global_params_present = global_params_node is not None
    ## The type's default cutoffs, with the yaml's values layered on top.
    global_constants = exp_type.resolve_constants(global_cfg)

    ## A Slice (ADR-0020): its clip, its own groups and its auto-removal
    ## cutoffs replace the whole recording's.  The whole recording keeps the
    ## caller's exclusion_group and adds the design's also_exclude.
    from . import slices as _slices

    the_slice = None
    also_groups = _slices.also_exclude(global_cfg)
    ## exclusion_group=None loads the design as written (the QC Viewer shows
    ## every chamber), but the experiment still names the group its removals
    ## are written to: the design's, as a Slice names its own.  Every load of
    ## the whole recording that names no group reads the design's, so the
    ## Hub, the QC Viewer and a script exclude the same chambers.
    apply_exclusions = exclusion_group is not None
    if exclusion_group is None or exclusion_group is DESIGN_GROUP:
        exclusion_group = _slices.whole_recording_group(global_cfg)
    if slice is not None:
        if not _slices.allows_slices(exp_type):
            raise _slices.SliceError(f"{path.name}: {_slices.refusal(exp_type)}")
        if tuple(float(v) for v in range_minutes) != (0.0, 0.0):
            raise ValueError("pass slice or range_minutes, not both: a Slice states its "
                             "own range")
        the_slice = _slices.get_slice(global_cfg, str(slice))
        range_minutes = the_slice.range_minutes
        exclusion_group = the_slice.exclusion_group
        also_groups = the_slice.also_exclude
        global_constants = {**global_constants, **dict(the_slice.constants)}
    experiment_type: str | None = (
        None if exp_type.is_custom else exp_type.name)
    factors_node = global_cfg.get("experimental_design_factors") or {}
    design_factors: list[str] = list(factors_node.keys()) if factors_node else []
    global_well_names: dict[str, str] = {
        str(k): str(v) for k, v in (global_cfg.get("well_names") or {}).items()
    }

    # Experiment-wide lick transformation toggle.  When unset the historical
    # default (True — apply the 0.25-power transform) is preserved.
    _tl_raw = global_cfg.get("transform_licks", True)
    transform_licks_default = bool(_tl_raw)

    dfm_nodes = cfg.get("dfms", cfg.get("DFMs", None))
    if dfm_nodes is None:
        raise ValueError("YAML must include `dfms:`.")

    # Support either:
    # - list form (requires '-' entries)
    # - mapping/dict form (no '-' entries), keyed by DFM id
    if isinstance(dfm_nodes, Mapping):
        items = []
        for k, v in dfm_nodes.items():
            if not isinstance(v, Mapping):
                raise ValueError("Each dfms{...} value must be an object/mapping.")
            node = dict(v)
            node.setdefault("id", int(k))
            items.append(node)
        dfm_nodes = items
    elif not isinstance(dfm_nodes, list):
        raise ValueError("`dfms` must be either a list or a mapping keyed by DFM id.")

    ## The Opto Program (data/Program.txt): read unless every DFM is
    ## `optogenetics: no`.  More than one such file stops the load — which one
    ## describes this recording is not a guess to make — and one that cannot be
    ## read is reported and set aside, since the program is optional.
    from . import opto_program as _opto

    opto_setting = _safe_setting(global_cfg.get(_opto.SETTING_KEY))
    dfm_opto_settings = [
        _safe_setting(n.get(_opto.SETTING_KEY)) for n in dfm_nodes
        if isinstance(n, Mapping) and n.get(_opto.SETTING_KEY) is not None]
    opto_program = None
    opto_program_error: str | None = None
    if opto_setting != _opto.SETTING_NO or any(
            v != _opto.SETTING_NO for v in dfm_opto_settings):
        program_path = _opto.find_program(data_dir)
        if program_path is not None:
            try:
                opto_program = _opto.read_program(program_path)
            except _opto.ProgramError as err:
                opto_program_error = str(err)
                print(f"  WARNING: {program_path.name} not used: {err}", flush=True)
            else:
                print(f"Read {program_path.name}: DFM section(s) "
                      f"{sorted(opto_program.dfms)}", flush=True)
                if opto_program.skipped:
                    print(f"  NOTE: {program_path.name}: {len(opto_program.skipped)} "
                          f"line(s) skipped or sections dropped (see summary.txt)",
                          flush=True)
                if opto_program.grammar == _opto.GRAMMAR_SIMPLE:
                    print(f"  NOTE: {program_path.name}: {_opto.SIMPLE_WARNING}",
                          flush=True)
    ## The light the type expects (ADR-0019): a Program Interval running a
    ## Paradigm type it does not take stops the load, like any other way
    ## the config fails its type.
    if opto_program is not None:
        dfm_ids = [int(n.get("id", n.get("ID"))) for n in dfm_nodes
                   if isinstance(n, Mapping) and n.get("id", n.get("ID")) is not None]
        light_problems = exp_type.light_problems(opto_program, dfm_ids or None)
        if light_problems and strict_type:
            raise ValueError(
                f"{path.name} does not satisfy experiment_type "
                f"'{exp_type.name}':\n  - " + "\n  - ".join(light_problems))
        for problem in light_problems:
            print(f"  WARNING: {problem}", flush=True)
    program_notes: list[str] = []
    resolved_nodes: list[dict] = []

    design = ExperimentDesign(experiment_type=experiment_type)

    # Read file-based exclusions once before the DFM loop: the own group and
    # every group it also excludes.  No group at all (exclusion_group=None)
    # loads the design as written.
    from .exclusions import read_exclusions as _read_exclusions
    file_excl_for_group: dict[int, list[int]] = {}
    applied_groups: list[str] = []
    if apply_exclusions:
        applied_groups = list(_slices.exclusion_groups(exclusion_group, also_groups))
        _all_file_excl = _read_exclusions(resolved_experiment_dir)
        for group in applied_groups:
            for dfm_id_, chambers_ in (_all_file_excl.get(group) or {}).items():
                merged_ = set(file_excl_for_group.get(dfm_id_, [])) | set(chambers_)
                file_excl_for_group[dfm_id_] = sorted(merged_)
        if file_excl_for_group:
            print(
                f"\nApplying exclusion group(s) {_slices.groups_label(applied_groups)} from "
                f"remove_chambers.csv:",
                flush=True,
            )

    # First pass: compute parameters and capture chamber assignments, but don't load data yet.
    dfm_specs: list[tuple[int, Parameters, dict[int, str], dict[str, str], dict[int, dict[str, str]]]] = []
    excluded_by_dfm: dict[int, list[int]] = {}
    for node in dfm_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Each dfms[] entry must be an object/mapping.")
        dfm_id = int(node.get("id", node.get("ID")))
        if is_environmental_monitor(dfm_id):
            raise ValueError(
                f"{path.name}: {ENVIRONMENTAL_MONITOR_MESSAGE}; remove id "
                f"{dfm_id} from dfms: (its DFM{dfm_id}_*.csv is not read)")

        dfm_params_node = node.get("params", node.get("parameters", None))
        dfm_overrides = _normalize_param_overrides(dfm_params_node)
        dfm_params_present = dfm_params_node is not None

        if not (global_params_present or dfm_params_present):
            raise ValueError(
                f"DFM {dfm_id} must define a `params` section either under global: or under the DFM entry."
            )

        ## Inside a Project only the physical keys may be overridden per DFM
        ## (ADR-0005): pi_direction and chamber_sets describe hardware, and
        ## already vary between DFMs of one recording. An analysis override
        ## here would bypass the Design authority one level down.
        if in_project:
            illegal = sorted(set(dfm_overrides) - PHYSICAL_DFM_KEYS)
            if illegal:
                raise ValueError(
                    f"DFM {dfm_id}: per-DFM params {illegal} are not allowed "
                    f"inside a Project — the project design owns them. Only "
                    f"{sorted(PHYSICAL_DFM_KEYS)} may vary per DFM."
                )

        problem = _opto.setting_problem(node.get(_opto.SETTING_KEY),
                                        where=f"dfms[{dfm_id}].{_opto.SETTING_KEY}")
        if problem and strict_type:
            raise ValueError(f"{path.name}: {problem}")
        if problem:
            print(f"  WARNING: {problem}", flush=True)

        # Precedence: defaults < global < dfm
        overrides = {**global_overrides, **dfm_overrides}

        ## chamber_size is owned by the Experiment Type via the Chamber Layout
        ## (ADR-0007) and derived, never read from the config. The old
        ## "must be explicitly specified" check and the type/size disagreement
        ## check it fed are both gone: the two can no longer disagree.
        overrides["chamber_size"] = derived_chamber_size
        base_size = derived_chamber_size
        base = Parameters.single_well() if base_size == 1 else Parameters.two_well()
        params = base.with_updates(**overrides)
        chambers_raw = node.get("chambers", node.get("Chambers"))
        if design_factors:
            chamber_assignments, chamber_factor_levels = _parse_chamber_factor_assignments(
                chambers_raw, design_factors
            )
        else:
            chamber_assignments = _parse_chamber_assignments(chambers_raw)
            chamber_factor_levels = {}

        ## What the Opto Program says and the entry leaves out (Progressive
        ## Ratio's paired chambers, from its trigger wells), filled in before
        ## the type's per-DFM checks run on the result.
        node, completion_notes = exp_type.complete_dfm_node(
            dfm_id, dict(node), params=params,
            dfm_program=(opto_program.section(dfm_id)
                         if opto_program is not None else None))
        for note in completion_notes:
            print(f"  NOTE: {note}", flush=True)
        program_notes.extend(completion_notes)
        resolved_nodes.append(dict(node))

        ## Per-DFM constraints the type imposes (Progressive Ratio's paired
        ## chambers and same-treatment chamber groups).  Checked against the
        ## config as written, before exclusions thin the assignments.
        dfm_problems = exp_type.validate_dfm(dfm_id, dict(node), chamber_assignments)
        if dfm_problems and strict_type:
            raise ValueError(
                f"{path.name} does not satisfy experiment_type "
                f"'{exp_type.name}':\n  - " + "\n  - ".join(dfm_problems))
        for problem in dfm_problems:
            print(f"  WARNING: {problem}", flush=True)

        # Warn about stale excluded_chambers in YAML (no longer applied).
        if node.get("excluded_chambers"):
            print(
                f"  WARNING: DFM {dfm_id} has 'excluded_chambers' in flic_config.yaml — "
                f"this key is no longer used. Migrate exclusions to remove_chambers.csv.",
                flush=True,
            )

        # Apply file-based exclusions for this DFM.
        excl_set = set(file_excl_for_group.get(dfm_id, []))
        if excl_set:
            chamber_assignments = {k: v for k, v in chamber_assignments.items() if k not in excl_set}
            chamber_factor_levels = {k: v for k, v in chamber_factor_levels.items() if k not in excl_set}
            excluded_by_dfm[dfm_id] = sorted(excl_set)
            print(
                f"  DFM {dfm_id}: excluding chamber(s) {sorted(excl_set)} "
                f"(group {_slices.groups_label(applied_groups)})",
                flush=True,
            )

        # well_names: DFM entry overrides global
        dfm_well_names_node = node.get("well_names") or {}
        well_names: dict[str, str] = {
            **global_well_names,
            **{str(k): str(v) for k, v in dfm_well_names_node.items()},
        }
        dfm_specs.append((dfm_id, params, chamber_assignments, well_names, chamber_factor_levels))

    # ── Cross-check config DFMs vs. data directory ──────────────────────
    config_ids = {s[0] for s in dfm_specs}

    # Discover DFM IDs present on disk (v3: DFM{id}_*.csv, v2: DFM_{id}[_*].csv).
    # DFM99_*.csv is the Environmental Monitor, never a DFM.
    data_ids: set[int] = set()
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if not f.suffix.lower() == ".csv":
                continue
            name = f.name
            # v3: DFM{id}_...; v2: DFM_{id}.csv or DFM_{id}_*.csv
            m = _re.match(r"DFM(\d+)_", name) or _re.match(r"DFM_(\d+)(?:_|\.)", name)
            if m and not is_environmental_monitor(m.group(1)):
                data_ids.add(int(m.group(1)))

    config_only = sorted(config_ids - data_ids)
    data_only = sorted(data_ids - config_ids)
    if config_only:
        import warnings
        warnings.warn(
            f"DFM(s) in config but missing from {data_label}: {config_only} — skipping these.",
            stacklevel=1,
        )
        print(f"  WARNING: DFM(s) in config but not in {data_label}: {config_only}",
              flush=True)
    if data_only:
        import warnings
        warnings.warn(
            f"DFM data file(s) in {data_label} but not in config: {data_only} — ignoring these.",
            stacklevel=1,
        )
        print(f"  WARNING: DFM(s) in {data_label} but not in config: {data_only}", flush=True)

    # Only load DFMs that are in both config and data.
    dfm_specs = [s for s in dfm_specs if s[0] in data_ids]
    if not dfm_specs:
        raise ValueError(
            "No DFMs to load: none of the configured DFM IDs have matching data files "
            f"in {data_dir}.  Config IDs: {sorted(config_ids)}, data IDs: {sorted(data_ids)}."
        )

    ## Program Facets (facet_source: program | program_cycles) are laid out
    ## from each DFM's Program.txt section, so every DFM loaded needs one.
    from . import program_facets as _facets

    why_none = opto_program_error
    if opto_program is None and why_none is None and opto_setting == _opto.SETTING_NO \
            and all(v == _opto.SETTING_NO for v in dfm_opto_settings):
        why_none = "optogenetics is no, so it is not read"
    facet_problems = _facets.program_problems(
        exp_type.resolve_facet_source(global_cfg), opto_program,
        [s[0] for s in dfm_specs], program_error=why_none)
    if facet_problems and strict_type:
        raise ValueError(f"{path.name}: " + "; ".join(facet_problems))
    for problem in facet_problems:
        print(f"  WARNING: {problem}", flush=True)

    n_total = len(dfm_specs)
    dfm_ids_str = ", ".join(str(s[0]) for s in dfm_specs)
    print(f"Loading {n_total} DFM(s) [{dfm_ids_str}] from {data_location.describe()}",
          flush=True)

    loaded: dict[int, DFM] = {}
    #: DFMs with no row in the range: where each recording ended.
    empty: dict[int, float] = {}
    if parallel and len(dfm_specs) > 1:
        if executor not in ("threads", "processes"):
            raise ValueError(f"executor must be 'threads' or 'processes', got {executor!r}")
        Exec = ThreadPoolExecutor if executor == "threads" else ProcessPoolExecutor
        with Exec(max_workers=max_workers) as pool:
            futs = {
                pool.submit(_load_dfm_for_config, dfm_id, params, data_dir, range_minutes): dfm_id
                for dfm_id, params, _, _wn, _fl in dfm_specs
            }
            first_error: RuntimeError | None = None
            first_exc: BaseException | None = None
            n_done = 0
            for fut in as_completed(futs):
                dfm_id = futs[fut]
                try:
                    result = fut.result()
                    if isinstance(result, EmptyRangeError):
                        empty[dfm_id] = result.recording_end
                    else:
                        loaded[dfm_id] = result
                    n_done += 1
                    print(f"  DFM {dfm_id} done  ({n_done}/{n_total})", flush=True)
                except Exception as e:  # noqa: BLE001
                    first_error = RuntimeError(f"Failed to load DFM {dfm_id} from YAML config.")
                    first_exc = e
                    # Cancel any futures that haven't started yet, then stop waiting.
                    for f in futs:
                        f.cancel()
                    break
        if first_error is not None:
            if first_exc is not None:
                raise first_error from first_exc
            raise first_error
    else:
        for n_done, (dfm_id, params, _, _wn, _fl) in enumerate(dfm_specs, 1):
            print(f"  Loading DFM {dfm_id}  ({n_done}/{n_total}) ...", flush=True)
            result = _load_dfm_for_config(dfm_id, params, data_dir, range_minutes)
            if isinstance(result, EmptyRangeError):
                empty[dfm_id] = result.recording_end
            else:
                loaded[dfm_id] = result
            print(f"  DFM {dfm_id} done", flush=True)

    ## A Slice is pooled only from Members that recorded all of it, every DFM
    ## reaching its end within the design's tolerance.
    if the_slice is not None:
        dfm_ends = {**empty, **{d: float(x.recording_end_minutes) for d, x in loaded.items()
                               if x.recording_end_minutes is not None}}
        reason = _slices.coverage(the_slice, dfm_ends, _slices.tolerance(global_cfg))
        if reason is not None:
            raise _slices.SliceNotCovered(the_slice, reason, dfm_ends,
                                          _slices.tolerance(global_cfg))
    elif empty:
        raise ValueError(
            f"No data in range {tuple(range_minutes)} for DFM(s) {sorted(empty)}: the "
            f"recording ends at minute {min(empty.values()):.1f}.")

    print(f"All {n_total} DFM(s) loaded.", flush=True)

    ## Rows under a ratio Paradigm counting Contact (PyMCU 4.0.1's default)
    ## hold ms of contact in Requirement and Progress, not Events, and a Well
    ## set On or Off reads Phase M with an Acclimation still to come; only
    ## Program.txt says which Paradigms and Wells those are.
    if opto_program is not None:
        contact = opto_program.contact_paradigms()
        for dfm_id, dfm in loaded.items():
            if dfm.paradigm is None:
                continue
            if contact:
                dfm.paradigm.mark_contact(contact)
            dfm.paradigm.mark_acclimation(opto_program.acclimation_rows(dfm_id, dfm.paradigm))

    # Store DFMs and assign chambers to treatments.
    chamber_factors_map: dict[tuple[int, int], dict[str, str]] = {}
    for dfm_id, _, chamber_assignments, well_names, chamber_factor_levels in dfm_specs:
        dfm = loaded[dfm_id]
        if well_names:
            dfm.well_names = well_names
        design.add_dfm(dfm, overwrite=True)
        for chamber_index, treatment_name in chamber_assignments.items():
            if treatment_name not in design.treatments:
                design.add_treatment(Treatment(treatment_name))
            design.treatments[treatment_name].add_chamber(dfm, chamber_index)
        for chamber_index, fl in chamber_factor_levels.items():
            chamber_factors_map[(int(dfm_id), int(chamber_index))] = fl

    ## Every DFM was built with the derived chamber_size, so they cannot
    ## disagree; the class follows from the Chamber Layout and, where the type
    ## brings analysis of its own, from the type.
    _CLS = _resolve_experiment_class(exp_type, chamber_layout)

    ## The resolved dfms: entries — what the Opto Program filled in included —
    ## are what a type reads back (Progressive Ratio's paired chambers).
    resolved_cfg = {k: v for k, v in cfg.items() if k not in ("global", "DFMs")}
    resolved_cfg["dfms"] = resolved_nodes
    exp = _CLS(
        dfms=design.dfms,
        design=design,
        global_config=global_cfg,
        global_constants=global_constants,
        well_names=global_well_names or None,
        design_factors=design_factors or None,
        chamber_factors=chamber_factors_map or None,
        config_path=path,
        data_dir=data_dir,
        data_location=data_location,
        experiment_dir=resolved_experiment_dir,
        config={"global": global_cfg, **resolved_cfg},
        experiment_type=exp_type,
        chamber_layout=chamber_layout,
        facet_cutoffs=exp_type.resolve_facet_cutoffs(global_cfg),
        range_minutes=(float(range_minutes[0]), float(range_minutes[1])),
        transform_licks=transform_licks_default,
        parallel=bool(parallel),
        executor=executor,
        max_workers=max_workers,
        optogenetics=opto_setting,
        opto_program=opto_program,
        opto_program_error=opto_program_error,
        program_notes=program_notes,
        slice=the_slice,
        also_exclude=tuple(applied_groups[1:]) if apply_exclusions else (),
    )

    if excluded_by_dfm:
        exp.excluded_chambers = excluded_by_dfm
    exp.exclusion_group = exclusion_group
    ## What Experiment.load_slice needs to load a sibling Slice the same way.
    exp.load_options = {
        "config_name": config_name, "parallel": parallel, "max_workers": max_workers,
        "executor": executor, "eager": eager, "use_disk_cache": use_disk_cache,
        "design_global": dict(design_global) if design_global else None,
        "in_project": in_project, "strict_type": strict_type,
    }
    if the_slice is not None:
        exp.slice_dfm_ends = {d: float(x.recording_end_minutes) for d, x in loaded.items()
                              if x.recording_end_minutes is not None}

    ## The disk cache holds the summary of the design as loaded, so its key
    ## carries everything that shaped it beyond the config file and the data:
    ## the design's global block (a Member inherits it from project.yaml), the
    ## chambers excluded and the Slice.
    cache_extra = _cache_signature(global_cfg, excluded_by_dfm, the_slice)

    if not eager:
        print("Skipping feeding-summary pre-compute (eager=False).", flush=True)
        return exp

    if use_disk_cache:
        from . import cache as _cache
        cached = _cache.load_feeding_summary(
            resolved_experiment_dir,
            range_minutes=(float(range_minutes[0]), float(range_minutes[1])),
            transform_licks=transform_licks_default,
            extra=cache_extra,
            data_location=data_location,
        )
        if cached is not None:
            print("Loaded feeding summary from disk cache.", flush=True)
            key = ((float(range_minutes[0]), float(range_minutes[1])), transform_licks_default)
            exp._feeding_summary_cache[key] = cached
            return exp

    print("Pre-computing feeding summary...", flush=True)
    df = exp.feeding_summary(range_minutes=(float(range_minutes[0]), float(range_minutes[1])))
    if use_disk_cache:
        try:
            _cache.save_feeding_summary(
                df, resolved_experiment_dir,
                range_minutes=(float(range_minutes[0]), float(range_minutes[1])),
                transform_licks=transform_licks_default,
                extra=cache_extra,
                data_location=data_location,
            )
        except Exception as e:  # pragma: no cover
            print(f"  (disk cache write skipped: {e})", flush=True)
    print("Ready.", flush=True)
    return exp


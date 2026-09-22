from __future__ import annotations

import re as _re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import yaml

from .dfm import DFM
from .experiment import Experiment
from .experiment_design import ExperimentDesign
from .hedonic_experiment import HedonicFeedingExperiment
from .parameters import Parameters
from .progressive_ratio_experiment import ProgressiveRatioExperiment
from .single_well_experiment import SingleWellExperiment
from .treatment import Treatment
from .two_well_experiment import TwoWellExperiment

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


def _load_dfm_for_config(
    dfm_id: int, params: Parameters, data_dir: str | Path, range_minutes: Sequence[float]
) -> DFM:
    # Separate top-level helper so process pools can pickle it.
    return DFM.load(dfm_id, params, data_dir=data_dir, range_minutes=range_minutes)


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
    exclusion_group: str | None = "general",
    design_global: Mapping[str, Any] | None = None,
    in_project: bool = False,
    strict_type: bool = True,
) -> Experiment:
    """
    Load an experiment from a experiment directory.

    Reads ``experiment_dir/<config_name>`` (default ``flic_config.yaml``) and
    loads DFM data from ``experiment_dir/data``.

    Parameters
    ----------
    experiment_dir:
        Project root directory.  Must contain the selected config file.
        Data is read from *experiment_dir/data*.  The returned ``Experiment``
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
    strict_type:
        Raise when the config violates its Experiment Type instead of warning.
        The linter passes False so it can report every problem at once.

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

    exp_type = _experiment_types.get_experiment_type(
        global_cfg.get("experiment_type"))
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

    data_dir = resolved_experiment_dir / "data"

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

    design = ExperimentDesign(experiment_type=experiment_type)

    # Read file-based exclusions once before the DFM loop.
    from .exclusions import read_exclusions as _read_exclusions
    file_excl_for_group: dict[int, list[int]] = {}
    if exclusion_group is not None:
        _all_file_excl = _read_exclusions(resolved_experiment_dir)
        file_excl_for_group = _all_file_excl.get(exclusion_group, {})
        if file_excl_for_group:
            print(
                f"\nApplying exclusion group '{exclusion_group}' from remove_chambers.csv:",
                flush=True,
            )

    # First pass: compute parameters and capture chamber assignments, but don't load data yet.
    dfm_specs: list[tuple[int, Parameters, dict[int, str], dict[str, str], dict[int, dict[str, str]]]] = []
    excluded_by_dfm: dict[int, list[int]] = {}
    for node in dfm_nodes:
        if not isinstance(node, Mapping):
            raise ValueError("Each dfms[] entry must be an object/mapping.")
        dfm_id = int(node.get("id", node.get("ID")))

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
                f"(group '{exclusion_group}')",
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

    # Discover DFM IDs present on disk (v3: DFM{id}_*.csv, v2: DFM_{id}[_*].csv)
    data_ids: set[int] = set()
    if data_dir.is_dir():
        for f in data_dir.iterdir():
            if not f.suffix.lower() == ".csv":
                continue
            name = f.name
            # v3: DFM{id}_...
            m = _re.match(r"DFM(\d+)_", name)
            if m:
                data_ids.add(int(m.group(1)))
                continue
            # v2: DFM_{id}.csv or DFM_{id}_*.csv
            m = _re.match(r"DFM_(\d+)(?:_|\.)", name)
            if m:
                data_ids.add(int(m.group(1)))

    config_only = sorted(config_ids - data_ids)
    data_only = sorted(data_ids - config_ids)
    if config_only:
        import warnings
        warnings.warn(
            f"DFM(s) in config but missing from data/: {config_only} — skipping these.",
            stacklevel=1,
        )
        print(f"  WARNING: DFM(s) in config but not in data/: {config_only}", flush=True)
    if data_only:
        import warnings
        warnings.warn(
            f"DFM data file(s) in data/ but not in config: {data_only} — ignoring these.",
            stacklevel=1,
        )
        print(f"  WARNING: DFM(s) in data/ but not in config: {data_only}", flush=True)

    # Only load DFMs that are in both config and data.
    dfm_specs = [s for s in dfm_specs if s[0] in data_ids]
    if not dfm_specs:
        raise ValueError(
            "No DFMs to load: none of the configured DFM IDs have matching data files "
            f"in {data_dir}.  Config IDs: {sorted(config_ids)}, data IDs: {sorted(data_ids)}."
        )

    n_total = len(dfm_specs)
    dfm_ids_str = ", ".join(str(s[0]) for s in dfm_specs)
    print(f"Loading {n_total} DFM(s) [{dfm_ids_str}] from {data_dir}", flush=True)

    loaded: dict[int, DFM] = {}
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
                    loaded[dfm_id] = fut.result()
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
            loaded[dfm_id] = _load_dfm_for_config(dfm_id, params, data_dir, range_minutes)
            print(f"  DFM {dfm_id} done", flush=True)

    print(f"All {n_total} DFM(s) loaded.", flush=True)

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

    exp = _CLS(
        dfms=design.dfms,
        design=design,
        global_config=global_cfg,
        global_constants=global_constants,
        well_names=global_well_names or None,
        design_factors=design_factors or None,
        chamber_factors=chamber_factors_map or None,
        config_path=path,
        experiment_dir=resolved_experiment_dir,
        config={"global": global_cfg, **{k: v for k, v in cfg.items() if k != "global"}},
        experiment_type=exp_type,
        chamber_layout=chamber_layout,
        facet_cutoffs=exp_type.resolve_facet_cutoffs(global_cfg),
        range_minutes=(float(range_minutes[0]), float(range_minutes[1])),
        transform_licks=transform_licks_default,
        parallel=bool(parallel),
        executor=executor,
        max_workers=max_workers,
    )

    if excluded_by_dfm:
        exp.excluded_chambers = excluded_by_dfm
    exp.exclusion_group = exclusion_group

    if not eager:
        print("Skipping feeding-summary pre-compute (eager=False).", flush=True)
        return exp

    if use_disk_cache:
        from . import cache as _cache
        cached = _cache.load_feeding_summary(
            resolved_experiment_dir,
            range_minutes=(float(range_minutes[0]), float(range_minutes[1])),
            transform_licks=transform_licks_default,
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
            )
        except Exception as e:  # pragma: no cover
            print(f"  (disk cache write skipped: {e})", flush=True)
    print("Ready.", flush=True)
    return exp


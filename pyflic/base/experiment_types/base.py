"""The ``ExperimentType`` (ADR-0007, ADR-0019).

An Experiment Type is a reusable definition of an assay's shape that selects
a **Chamber Layout** and constrains the rest of an experiment — the required
``well_names``, the facet cutoffs and phase labels, the default
``constants:``, the light it expects, the **Analysis Modules** that run, the
Plot Specs of its report, and the report produced.  It is a *composed
strategy object*, not an ``Experiment`` subclass: the subclass hierarchy
stays where it belongs, one level down, expressing the data shape, and a
module that carries maths of its own names its subclass.

Since ADR-0019 a type is data: one instance of this class built from a
:class:`~pyflic.base.experiment_types.definition.TypeDefinition` — a
shipped one (Custom, Hedonic, ProgressiveRatio) or a lab's own — whose
modules supply the behaviour that used to be subclass code.  Per ADR-0007 a
type **owns** its fixed fields — a typed ``flic_config.yaml`` omits
``chamber_layout`` and ``params.chamber_size`` and they are derived here,
never written to disk.  Built from no definition, the class is the
permissive default.
"""

from __future__ import annotations

from .. import opto_program as op
from .. import program_facets, windowing

#: Chamber Layout -> the ``params.chamber_size`` it implies.
LAYOUT_CHAMBER_SIZE: dict[str, int] = {"single_well": 1, "two_well": 2}
LAYOUTS = tuple(LAYOUT_CHAMBER_SIZE)

#: The ``global:`` key naming a Treatment control's control level (ADR-0019).
CONTROL_LEVEL_KEY = "control_level"


class ExperimentType:
    """A type built from a definition; with none, the permissive default."""

    #: Canonical key stored in the yaml (``experiment_type:``).  Case-insensitive.
    name: str = "Custom"
    #: Human label for menus, the report cover, dialogs.
    display_name: str = "Custom"

    #: Fixed Chamber Layout, or ``None`` to read it from the yaml (Custom).
    chamber_layout: str | None = None
    #: Well keys the type requires ``well_names`` to name, or ``None`` for no
    #: constraint.  A two-well type needs both wells named or its figures have
    #: no axis labels worth printing.
    required_wells: tuple[str, ...] | None = None
    #: Facet cutoffs the type suggests.  By default only a *default* (the user
    #: may change it); set ``facets_fixed`` to make it immovable.  ``None``
    #: means "read entirely from the yaml" (Custom).
    facet_cutoffs: tuple[float, ...] | None = None
    #: When True the type owns ``facet_cutoffs`` outright.
    facets_fixed: bool = False
    #: When True the type's Facets are not minute cutoffs at all but windows
    #: the *data* defines (Progressive Ratio splits at each Chamber Group's own
    #: training end, ADR-0013).  ``facet_cutoffs`` is then owned and absent,
    #: and consumers group rows by the ``Facet`` label rather than by
    #: ``FacetRange`` — as they do for Program Facets (``facet_source``); ask
    #: :meth:`facets_vary_by_dfm`, which covers both.
    data_derived_facets: bool = False
    #: Names for the phases of the *default* facet structure; ``len`` should
    #: equal ``len(facet_cutoffs) + 1``.  Applied only when the actual cutoffs
    #: match the default (see :meth:`phase_labels_for`).
    phase_labels: tuple[str, ...] = ()
    #: Dotted path of the ``Experiment`` subclass this type instantiates, or
    #: ``None`` to derive it from the Chamber Layout alone.  A type names one
    #: only when it brings analysis methods of its own (Hedonic's weighted
    #: durations, Progressive Ratio's breaking point) — the strategy object
    #: carries the constraints, the subclass carries the maths.
    experiment_class: str | None = None

    #: Defaults merged under the yaml's ``constants:`` block — the auto-removal
    #: cutoffs.  A yaml value always wins.
    default_constants: dict = {}

    #: The definition the type was built from, or ``None``.
    definition = None
    #: Where the registry found it: ``"shipped"``, ``"copy"`` (beside the
    #: data) or ``"library"``; ``None`` when built directly.
    resolved_from: str | None = None
    #: Its Analysis Modules, in the order :data:`analysis_modules.MODULES`
    #: lists them.
    modules: tuple = ()

    def __init__(self, definition=None) -> None:
        if definition is not None:
            self._apply(definition)

    def _apply(self, definition) -> None:
        """Take every attribute the definition and its modules decide."""
        from ..analysis_modules import MODULES

        self.definition = definition
        self.modules = tuple(m for m in MODULES if m.id in definition.modules)
        self.name = definition.name
        self.display_name = definition.label
        self.chamber_layout = definition.chamber_layout
        wells = list(definition.required_wells)
        for module in self.modules:
            wells += [w for w in module.required_wells if w not in wells]
        self.required_wells = tuple(wells) or None
        derived = next((m for m in self.modules if m.data_derived_facets), None)
        if derived is not None:
            self.facet_cutoffs = None
            self.facets_fixed = True
            self.data_derived_facets = True
            self.phase_labels = tuple(derived.phase_labels)
        else:
            self.facet_cutoffs = definition.facet_cutoffs
            self.facets_fixed = definition.facets_fixed
            self.data_derived_facets = False
            self.phase_labels = tuple(definition.facet_labels)
        self.experiment_class = next(
            (m.experiment_class for m in self.modules if m.experiment_class), None)
        ## The signal-ceiling cutoff is every type's (pyflic.base.ceiling): a
        ## sensor at the top of its range shows no lick whatever is analysed.
        from ..ceiling import DEFAULT_CONSTANTS as CEILING_DEFAULTS

        constants: dict = dict(CEILING_DEFAULTS)
        for module in self.modules:
            constants.update(module.default_constants)
        constants.update(definition.constants)
        self.default_constants = constants

    # ---- identity -----------------------------------------------------

    @property
    def is_custom(self) -> bool:
        """Whether this is the Custom Experiment: the type a config with no
        ``experiment_type`` key has."""
        return self.name == "Custom"

    @property
    def layout_free(self) -> bool:
        """Whether the config states the Chamber Layout (``chamber_layout:``)
        because the type leaves it open."""
        return self.chamber_layout is None

    @property
    def is_shipped(self) -> bool:
        """Whether pyflic ships the type (read-only), rather than a lab."""
        return self.definition is None or bool(self.definition.shipped)

    @property
    def module_ids(self) -> tuple[str, ...]:
        return tuple(m.id for m in self.modules)

    def has_module(self, module_id: str) -> bool:
        """Whether the type includes the Analysis Module *module_id*."""
        return module_id in self.module_ids

    @property
    def light(self):
        """The light the type expects: ``"none"``, ``"any"`` or a tuple of
        Paradigm types (``definition.LIGHT_*``); ``"any"`` without a
        definition."""
        return "any" if self.definition is None else self.definition.light

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.name}>"

    # ---- derivation (the type owns these) -----------------------------

    def resolve_chamber_layout(self, global_cfg: dict | None) -> str:
        """The Chamber Layout to use: fixed by the type, else read from the yaml.

        A typed config that states ``chamber_layout`` anyway is not an error
        here — :meth:`validate` reports it — so loading never depends on the
        order the two checks run in.
        """
        if self.chamber_layout is not None:
            return self.chamber_layout
        raw = str((global_cfg or {}).get("chamber_layout") or "").strip().lower()
        raw = raw.replace("-", "_").replace(" ", "_")
        if not raw:
            return "two_well"
        if raw not in LAYOUT_CHAMBER_SIZE:
            raise ValueError(
                f"Unknown chamber_layout {raw!r}. "
                f"Valid values: {', '.join(LAYOUTS)}."
            )
        return raw

    def resolve_chamber_size(self, global_cfg: dict | None) -> int:
        """The ``params.chamber_size`` implied by the Chamber Layout."""
        return LAYOUT_CHAMBER_SIZE[self.resolve_chamber_layout(global_cfg)]

    def resolve_facet_cutoffs(self, global_cfg: dict | None):
        """The facet cutoffs to use, or ``None`` when the experiment is not
        faceted (a Custom Experiment that never set any) or its Facets come
        from the Opto Program (``facet_source``)."""
        if self.facets_fixed and self.facet_cutoffs is not None:
            return tuple(self.facet_cutoffs)
        if self.data_derived_facets:
            ## The windows come from the data, per Chamber Group; there is no
            ## cutoff list to hand back and the yaml is not consulted.
            return None
        if program_facets.is_program_source(self.resolve_facet_source(global_cfg)):
            ## Each DFM's Program Intervals are its Facets; a type's default
            ## cutoffs would split the recording a second, unrelated way.
            return None
        raw = (global_cfg or {}).get("facet_cutoffs")
        if raw is not None:
            return tuple(windowing.normalize_cutoffs(raw))
        return tuple(self.facet_cutoffs) if self.facet_cutoffs is not None else None

    @property
    def owns_facet_source(self) -> bool:
        """A type that owns its Facets (fixed cutoffs, or windows from the
        data) owns ``facet_source`` too (decision 7)."""
        return bool(self.facets_fixed or self.data_derived_facets)

    def resolve_facet_source(self, global_cfg: dict | None) -> str:
        """Where the Facets come from: ``cutoffs`` (the default), ``program``
        or ``program_cycles`` (:mod:`pyflic.base.program_facets`).  A type
        that owns its Facets is always ``cutoffs`` here; an unreadable value
        is too, and :meth:`validate` reports it."""
        if self.owns_facet_source:
            return program_facets.SOURCE_CUTOFFS
        try:
            return program_facets.normalize_source(
                (global_cfg or {}).get(program_facets.FACET_SOURCE_KEY))
        except ValueError:
            return program_facets.SOURCE_CUTOFFS

    def facets_vary_by_dfm(self, global_cfg: dict | None) -> bool:
        """Whether a Facet's windows can differ between DFMs, so consumers
        must group rows by the ``Facet`` label rather than by ``FacetRange``:
        windows from the data (Progressive Ratio) or from each DFM's program."""
        return bool(self.data_derived_facets) or program_facets.is_program_source(
            self.resolve_facet_source(global_cfg))

    def resolve_constants(self, global_cfg: dict | None) -> dict:
        """The effective ``constants:`` block — the type's defaults with the
        yaml's values layered on top.

        A yaml value of ``null`` wins over a type default and means that
        auto-filter check is off (the editors write ``null`` when a cutoff's
        checkbox is unchecked).  An omitted key still inherits the type's
        default.
        """
        merged = dict(self.default_constants)
        for key, value in ((global_cfg or {}).get("constants") or {}).items():
            merged[key] = value
        return merged

    def phase_labels_for(self, windows, global_cfg: dict | None) -> list[str]:
        """Display labels for *windows*.

        The type's :attr:`phase_labels` apply only while the cutoffs are the
        type's default and the counts line up; any other windowing gets plain
        minute-range labels, because "Acclimation" would be a lie once the user
        has moved the cutoff that defined it.
        """
        explicit = (global_cfg or {}).get("facet_labels")
        if explicit and len(list(explicit)) == len(windows):
            return [str(v) for v in explicit]
        cutoffs = self.resolve_facet_cutoffs(global_cfg)
        default = tuple(self.facet_cutoffs) if self.facet_cutoffs is not None else None
        matches_default = (
            default is not None
            and cutoffs is not None
            and tuple(float(c) for c in cutoffs) == tuple(float(c) for c in default)
        )
        if matches_default and len(self.phase_labels) == len(windows):
            return list(self.phase_labels)
        return [windowing.minute_label(w) for w in windows]

    def primary_phase_index(self, windows) -> int:
        """Index of the window the headline result is read from: the second
        when there are two or more, else the only one."""
        return 1 if len(windows) >= 2 else 0

    # ---- constraints --------------------------------------------------

    #: Keys a typed config must NOT state, because the type owns them.
    def owned_keys(self) -> tuple[str, ...]:
        owned: list[str] = []
        if self.chamber_layout is not None:
            owned.append("chamber_layout")
        if self.facets_fixed or self.data_derived_facets:
            owned.append("facet_cutoffs")
        if self.owns_facet_source:
            owned.append(program_facets.FACET_SOURCE_KEY)
        return tuple(owned)

    def validate_dfm(self, dfm_id: int, node: dict | None,
                     chamber_assignments: dict | None) -> list[str]:
        """Problems with one ``dfms:`` entry under this type.

        *chamber_assignments* is the parsed ``{chamber: treatment}`` mapping as
        the config states it (before any exclusion is applied).  The type has
        no per-DFM constraints of its own; a module with a chamber-level
        structure (the Ratio analysis's paired/yoked roles) checks it here.
        Never raises.
        """
        out: list[str] = []
        for module in self.modules:
            out += module.validate_dfm(dfm_id, node, chamber_assignments)
        return out

    def complete_dfm_node(self, dfm_id: int, node: dict, *, params=None,
                          dfm_program=None) -> tuple[dict, list[str]]:
        """``(node, notes)``: one ``dfms:`` entry with what the Opto Program
        implies and the entry leaves out filled in, and one note per thing
        taken from the program or contradicting it.

        *params* is the DFM's resolved :class:`~pyflic.base.parameters.Parameters`
        and *dfm_program* its :class:`~pyflic.base.opto_program.DFMProgram`, or
        ``None`` without a ``Program.txt`` section.  The type takes nothing
        from the program; the Ratio analysis derives ``paired_chambers`` from
        its trigger wells.  Never raises.
        """
        notes: list[str] = []
        for module in self.modules:
            node, more = module.complete_dfm_node(dfm_id, node, params=params,
                                                  dfm_program=dfm_program)
            notes += more
        return node, notes

    def report_facets(self) -> list[str] | None:
        """Facet labels the type's report figures show by default, or ``None``
        for every Facet.  A default, not a gate — the Plot Editor can still
        name any Facet."""
        for module in self.modules:
            if module.report_facets is not None:
                return list(module.report_facets)
        return None

    # ---- the light ----------------------------------------------------

    def light_problems(self, program, dfm_ids=None) -> list[str]:
        """Where *program* (an :class:`~pyflic.base.opto_program.OptoProgram`)
        runs light the type does not expect, for *dfm_ids* (default every
        section), as one line per Program Interval (ADR-0019).

        A lit interval runs its Paradigm's type for its level Wells and
        OpenLoop for its ``On`` Wells; an interval whose Wells are all
        ``Off`` runs nothing, whatever its Paradigm (PyMCU's ``Idle`` is a
        ClosedLoop).  A type that expects no light takes none; one that
        expects a set of Paradigm types takes those, with OpenLoop beside
        them as a control; ``any`` takes everything.  Never raises.
        """
        light = self.light
        if program is None or light == "any":
            return []
        if getattr(program, "grammar", None) == op.GRAMMAR_SIMPLE:
            return []
        expected = set(light) | {op.TYPE_NAMES[op.OPEN_LOOP]} if isinstance(light, tuple) \
            else set()
        ids = sorted(program.dfms) if dfm_ids is None else sorted({int(d) for d in dfm_ids})
        found: dict[str, list[str]] = {}
        for dfm_id in ids:
            section = program.section(dfm_id)
            if section is None:
                continue
            for number, iv in enumerate(section.intervals, start=1):
                kinds: set[str] = set()
                if iv.open_wells():
                    kinds.add(op.TYPE_NAMES[op.OPEN_LOOP])
                if iv.trigger_wells():
                    kinds.add(op.TYPE_NAMES.get(iv.paradigm.type, iv.paradigm.type_name))
                for kind in sorted(kinds - expected):
                    found.setdefault(kind, []).append(f"DFM {dfm_id} interval {number}")
        if not found:
            return []
        if light == "none":
            wanted = "expects no light"
        else:
            wanted = (f"expects {', '.join(light)}, with OpenLoop and dark intervals "
                      f"beside them")
        return [f"the Opto Program runs {kind} ({'; '.join(where)}), and experiment_type "
                f"'{self.name}' {wanted}; choose or build a type that expects it"
                for kind, where in found.items()]

    def validate(self, global_cfg: dict | None) -> list[str]:
        """Problems with *global_cfg* under this type, as human-readable lines.

        Never raises — the caller decides whether a problem is fatal, so the
        linter can report every problem at once instead of the first.
        """
        problems: list[str] = []
        g = global_cfg or {}
        for key in self.owned_keys():
            if g.get(key) is not None:
                problems.append(
                    f"'{key}' is owned by experiment_type '{self.name}' and must "
                    f"not appear in the config (remove it; the value is derived)"
                )
        params = g.get("params") or {}
        if self.chamber_layout is not None and params.get("chamber_size") is not None:
            expected = LAYOUT_CHAMBER_SIZE[self.chamber_layout]
            problems.append(
                f"'params.chamber_size' is owned by experiment_type "
                f"'{self.name}' (chamber_layout '{self.chamber_layout}' implies "
                f"{expected}) and must not appear in the config"
            )
        key = program_facets.FACET_SOURCE_KEY
        if not self.owns_facet_source:
            problem = program_facets.source_problem(g.get(key), where=f"global.{key}")
            if problem:
                problems.append(problem)
            elif program_facets.is_program_source(self.resolve_facet_source(g)) \
                    and g.get("facet_cutoffs") is not None:
                problems.append(
                    f"'facet_cutoffs' and '{key}: {self.resolve_facet_source(g)}' "
                    f"cannot both be stated: Program Facets come from each DFM's "
                    f"Program Intervals, not from minute cutoffs (remove one)")
        if self.required_wells:
            named = {str(k).strip().upper()
                     for k, v in (g.get("well_names") or {}).items()
                     if str(v or "").strip()}
            missing = [w for w in self.required_wells if w.upper() not in named]
            if missing:
                problems.append(
                    f"experiment_type '{self.name}' requires well_names for "
                    f"{', '.join(missing)}"
                )
        ## The optogenetic light QC applies to every type, so its setting and
        ## its constants are checked here, once, for all of them.
        from ..opto_program import SETTING_KEY, setting_problem
        from ..opto_qc import opto_constant_problems

        problem = setting_problem(g.get(SETTING_KEY), where=f"global.{SETTING_KEY}")
        if problem:
            problems.append(problem)
        problems += opto_constant_problems(g.get("constants"))
        ## So is the signal-ceiling cutoff.
        from ..ceiling import ceiling_problems

        problems += ceiling_problems(g.get("constants"))
        for module in self.modules:
            problems += module.validate(g)
        problems += self._design_problems(g)
        return problems

    def _design_problems(self, g: dict) -> list[str]:
        """The type's factor names and its Treatment control (ADR-0019):
        the config declares every factor the type names, and may add more;
        a Treatment control's level is one of its factor's levels."""
        if self.definition is None:
            return []
        problems: list[str] = []
        declared = g.get("experimental_design_factors") or {}
        if not isinstance(declared, dict):
            declared = {}
        missing = [f for f in self.definition.factors if f not in declared]
        if missing:
            problems.append(
                f"experiment_type '{self.name}' needs the experimental design factor"
                f"{'s' if len(missing) > 1 else ''} {', '.join(missing)} declared")
        factor = self.definition.treatment_control
        level = g.get(CONTROL_LEVEL_KEY)
        if factor is None:
            if level is not None:
                problems.append(
                    f"'{CONTROL_LEVEL_KEY}' is set, and experiment_type '{self.name}' has "
                    f"no Treatment control (remove it)")
        elif level is None or str(level).strip() == "":
            problems.append(
                f"experiment_type '{self.name}' compares every {factor} with a control: "
                f"name it with '{CONTROL_LEVEL_KEY}', one of the levels of {factor}")
        elif factor in declared and str(level) not in [str(v) for v in
                                                       (declared.get(factor) or [])]:
            problems.append(
                f"'{CONTROL_LEVEL_KEY}' {level!r} is not a level of {factor} "
                f"({', '.join(str(v) for v in declared.get(factor) or [])})")
        return problems

    def control_level(self, global_cfg: dict | None) -> str | None:
        """The Treatment control's level the config names, or ``None``."""
        if self.definition is None or self.definition.treatment_control is None:
            return None
        level = (global_cfg or {}).get(CONTROL_LEVEL_KEY)
        return None if level is None or str(level).strip() == "" else str(level)

    # ---- outputs ------------------------------------------------------

    def report_intro(self) -> str:
        """One paragraph for the report cover."""
        if self.definition is not None and self.definition.description:
            return self.definition.description
        return ("A FLIC feeding experiment analysed without a specific "
                "experiment type, so no type-level constraints were applied.")

    def report_set(self, chamber_layout: str) -> list[str]:
        """Ordered Plot Spec ids the type's report and the Standard Pipeline
        produce with no authoring: its modules' first, then the built-in
        plots it picks, then those it has a Plot Spec of its own for, each
        once and only where *chamber_layout* can draw it (a preference index
        needs two wells).  A **default**, never a gate — any action the
        layout and type permit stays available in the Script Editor.
        """
        if self.definition is None:
            picked = ["faceted_pi", "faceted_licks", "faceted_events",
                      "faceted_medduration", "timecourse_pi", "timecourse_licks"]
            ids = picked
        else:
            ids = [p for m in self.modules for p in m.plots]
            ids += list(self.definition.plots) + list(self.definition.plot_specs)
        from ..pubfigures import plot_allowed

        out: list[str] = []
        for plot_id in ids:
            if plot_id not in out and plot_allowed(plot_id, chamber_layout, self.module_ids):
                out.append(plot_id)
        return out

    def plot_specs(self) -> dict:
        """The type's own Plot Specs, by plot id, as ``plot_specs.yaml``
        holds them under ``plots:``: what a new Project of the type starts
        with (ADR-0019)."""
        if self.definition is None:
            return {}
        return {k: dict(v) if isinstance(v, dict) else v
                for k, v in self.definition.plot_specs.items()}

    def output_manifest(self) -> list[str]:
        """Analysis outputs a run of this type is expected to leave in
        ``analysis/``.  Used by status displays to answer "has this been
        analysed?" without loading any data."""
        out = ["feeding_summary.csv", "feeding_summary_facet.csv", "summary.txt"]
        for module in self.modules:
            out += [f for f in module.outputs if f not in out]
        return out

    def project_results_blocks(self, project) -> list:
        """Pooled figures of the type's own for the Project Report's Results,
        after the report set (``report_layout`` blocks, from the Project's
        saved Combined Analysis): its modules', in turn.  None by default."""
        blocks: list = []
        for module in self.modules:
            blocks += module.project_results_blocks(self, project)
        return blocks

    # ---- scaffolding --------------------------------------------------

    def build_global(self, *, params: dict | None = None,
                     well_names: dict | None = None,
                     constants: dict | None = None,
                     factors: dict | None = None,
                     facet_cutoffs=None,
                     transform_licks: bool | None = None,
                     facet_source: str | None = None,
                     chamber_layout: str | None = None) -> dict:
        """A ``global:`` block for a fresh experiment of this type.

        Keys the type owns are deliberately absent — that is the whole point of
        ADR-0007's "derived, never written to disk".  A type that leaves the
        layout open writes *chamber_layout* (``two_well`` by default).  A
        *facet_source* of ``program`` or ``program_cycles`` is written in
        place of any cutoffs; ``cutoffs``, the default, is never written.
        """
        g: dict = {}
        if not self.is_custom:
            g["experiment_type"] = self.name
        if self.layout_free:
            g["chamber_layout"] = chamber_layout or "two_well"
        if transform_licks is not None:
            g["transform_licks"] = bool(transform_licks)
        source = (program_facets.SOURCE_CUTOFFS if self.owns_facet_source
                  else program_facets.normalize_source(facet_source))
        cutoffs = facet_cutoffs if facet_cutoffs is not None else self.facet_cutoffs
        if program_facets.is_program_source(source):
            g[program_facets.FACET_SOURCE_KEY] = source
        elif cutoffs is not None and not self.facets_fixed \
                and not self.data_derived_facets:
            g["facet_cutoffs"] = [_clean_number(v) for v in cutoffs]
        clean_params = dict(params or {})
        clean_params.pop("chamber_size", None)
        if clean_params:
            g["params"] = clean_params
        if well_names:
            g["well_names"] = dict(well_names)
        ## Only what the caller stated — including explicit ``null`` to turn a
        ## typed cutoff off.  Materialising ``default_constants`` here would
        ## freeze every design against today's numbers (ADR-0011); load-time
        ## ``resolve_constants()`` is what applies the type's defaults.
        if constants:
            g["constants"] = dict(constants)
        if factors:
            g["experimental_design_factors"] = {
                str(k): [str(v) for v in (vals or [])]
                for k, vals in factors.items()
            }
        return g


def _clean_number(value):
    """Whole numbers as ints (70.0 -> 70) so the yaml stays tidy."""
    try:
        return int(value) if float(value) == int(value) else float(value)
    except (TypeError, ValueError):
        return value

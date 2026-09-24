"""The ``ExperimentType`` base class (ADR-0007).

An Experiment Type is a named bundle that selects one **Chamber Layout** and
constrains the rest of an experiment — the required ``well_names``, the facet
cutoffs and phase labels, the default ``constants:``, the analyses that run, the
plot set, and the report produced.  It is a *composed strategy object*, not an
``Experiment`` subclass: the subclass hierarchy stays where it belongs, one
level down, expressing the data shape.

Per ADR-0007 a type **owns** its fixed fields — a typed ``flic_config.yaml``
omits ``chamber_layout`` and ``params.chamber_size`` and they are derived here,
never written to disk.  The base class is the permissive default, which is what
a Custom Experiment uses directly.
"""

from __future__ import annotations

from .. import windowing

#: Chamber Layout -> the ``params.chamber_size`` it implies.
LAYOUT_CHAMBER_SIZE: dict[str, int] = {"single_well": 1, "two_well": 2}
LAYOUTS = tuple(LAYOUT_CHAMBER_SIZE)


class ExperimentType:
    """Permissive base type.  A Custom Experiment uses this behaviour directly."""

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
    #: ``FacetRange``.
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

    # ---- identity -----------------------------------------------------

    @property
    def is_custom(self) -> bool:
        return self.chamber_layout is None

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
        faceted (a Custom Experiment that never set any)."""
        if self.facets_fixed and self.facet_cutoffs is not None:
            return tuple(self.facet_cutoffs)
        if self.data_derived_facets:
            ## The windows come from the data, per Chamber Group; there is no
            ## cutoff list to hand back and the yaml is not consulted.
            return None
        raw = (global_cfg or {}).get("facet_cutoffs")
        if raw is not None:
            return tuple(windowing.normalize_cutoffs(raw))
        return tuple(self.facet_cutoffs) if self.facet_cutoffs is not None else None

    def resolve_constants(self, global_cfg: dict | None) -> dict:
        """The effective ``constants:`` block — the type's defaults with the
        yaml's values layered on top."""
        merged = dict(self.default_constants)
        merged.update((global_cfg or {}).get("constants") or {})
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
        return tuple(owned)

    def validate_dfm(self, dfm_id: int, node: dict | None,
                     chamber_assignments: dict | None) -> list[str]:
        """Problems with one ``dfms:`` entry under this type.

        *chamber_assignments* is the parsed ``{chamber: treatment}`` mapping as
        the config states it (before any exclusion is applied).  The base type
        has no per-DFM constraints; a type with a chamber-level structure
        (Progressive Ratio's paired/yoked roles) checks it here.  Never raises.
        """
        return []

    def complete_dfm_node(self, dfm_id: int, node: dict, *, params=None,
                          dfm_program=None) -> tuple[dict, list[str]]:
        """``(node, notes)``: one ``dfms:`` entry with what the Opto Program
        implies and the entry leaves out filled in, and one note per thing
        taken from the program or contradicting it.

        *params* is the DFM's resolved :class:`~pyflic.base.parameters.Parameters`
        and *dfm_program* its :class:`~pyflic.base.opto_program.DFMProgram`, or
        ``None`` without a ``Program.txt`` section.  The base type takes
        nothing from the program; Progressive Ratio derives ``paired_chambers``
        from its trigger wells.  Never raises.
        """
        return node, []

    def report_facets(self) -> list[str] | None:
        """Facet labels the type's report figures show by default, or ``None``
        for every Facet.  A default, not a gate — the Plot Editor can still
        name any Facet."""
        return None

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
        return problems

    # ---- outputs ------------------------------------------------------

    def report_intro(self) -> str:
        """One paragraph for the report cover."""
        return ("A FLIC feeding experiment analysed without a specific "
                "experiment type, so no type-level constraints were applied.")

    def report_set(self, chamber_layout: str) -> list[str]:
        """Ordered Plot Spec ids the type's report and the Standard Pipeline
        produce with no authoring.  A **default**, never a gate — any action the
        layout and type permit stays available in the Script Editor.
        """
        if chamber_layout == "two_well":
            return ["faceted_pi", "faceted_licks", "faceted_events",
                    "faceted_medduration", "timecourse_pi", "timecourse_licks"]
        return ["faceted_licks", "faceted_events", "faceted_medduration",
                "timecourse_licks"]

    def output_manifest(self) -> list[str]:
        """Analysis outputs a run of this type is expected to leave in
        ``analysis/``.  Used by status displays to answer "has this been
        analysed?" without loading any data."""
        return ["feeding_summary.csv", "feeding_summary_facet.csv", "summary.txt"]

    def project_results_blocks(self, project) -> list:
        """Pooled figures of the type's own for the Project Report's Results,
        after the report set (``report_layout`` blocks, from the Project's
        saved Combined Analysis).  None by default."""
        return []

    # ---- scaffolding --------------------------------------------------

    def build_global(self, *, params: dict | None = None,
                     well_names: dict | None = None,
                     constants: dict | None = None,
                     factors: dict | None = None,
                     facet_cutoffs=None,
                     transform_licks: bool | None = None) -> dict:
        """A ``global:`` block for a fresh experiment of this type.

        Keys the type owns are deliberately absent — that is the whole point of
        ADR-0007's "derived, never written to disk".
        """
        g: dict = {}
        if self.name != "Custom":
            g["experiment_type"] = self.name
        elif self.chamber_layout is None:
            g["chamber_layout"] = "two_well"
        if transform_licks is not None:
            g["transform_licks"] = bool(transform_licks)
        cutoffs = facet_cutoffs if facet_cutoffs is not None else self.facet_cutoffs
        if cutoffs is not None and not self.facets_fixed \
                and not self.data_derived_facets:
            g["facet_cutoffs"] = [_clean_number(v) for v in cutoffs]
        clean_params = dict(params or {})
        clean_params.pop("chamber_size", None)
        if clean_params:
            g["params"] = clean_params
        if well_names:
            g["well_names"] = dict(well_names)
        merged_constants = dict(self.default_constants)
        merged_constants.update(constants or {})
        if merged_constants:
            g["constants"] = merged_constants
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

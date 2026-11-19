"""An Experiment Type's definition: the yaml a lab builds (ADR-0019).

An Experiment Type is data, not a class: a reusable definition of an assay's
shape that combines the **Analysis Modules** pyflic provides
(:mod:`pyflic.base.analysis_modules`) and never creates one.  The shipped
types (Custom, Hedonic, ProgressiveRatio) are definitions too, read-only, in
``experiment_types/shipped/``; a lab's own are built in the Experiment Type
Editor, kept in the lab library and copied beside the data that uses them
(:mod:`pyflic.base.experiment_types.library`).

The file, every key but ``name`` optional::

    format: 1
    name: HedonicClosedLoop             # what experiment_type: names; unique
    display_name: Hedonic Feeding, closed loop
    description: A two-well choice under closed-loop light ...  # the report intro
    chamber_layout: two_well            # single_well | two_well | either
    light: [ClosedLoop]                 # none | any | a list of Paradigm types
    required_wells: [A, B]
    factors: [Genotype]                 # Experimental Design Factor names
    modules: [hedonic]                  # Analysis Module ids
    control:
      roles: [Paired, Yoked]            # a Role comparison
      treatment: Genotype               # a Treatment control: one of factors
    plots: [faceted_pi, faceted_licks]  # built-in Plot Spec ids for the report
    plot_specs: {}                      # the type's own Plot Specs, by id
    facets:
      cutoffs: [30]                     # default minute cutoffs
      fixed: false                      # true: the config may not change them
      labels: [Acclimation, Test]       # the default windows' names
    constants: {max_events_cutoff: 1000}

Pure: no Qt and no plotting imports, so the loader, ``pyflic lint`` and the
editor read the same rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from .. import opto_program as op
from .. import opto_roles

#: The definition format this pyflic writes and reads.
FORMAT = 1

#: ``chamber_layout:`` values; ``either`` leaves the layout to the config.
LAYOUT_EITHER = "either"
DEFINITION_LAYOUTS: tuple[str, ...] = ("single_well", "two_well", LAYOUT_EITHER)

#: ``light:`` values other than a list of Paradigm types.
LIGHT_NONE, LIGHT_ANY = "none", "any"

#: The Paradigm types a ``light:`` list may name, in PyMCU's spelling.
PARADIGM_TYPE_NAMES: tuple[str, ...] = tuple(op.TYPE_NAMES[t] for t in op.PARADIGM_TYPES)
_PARADIGM_BY_KEY = {n.lower(): n for n in PARADIGM_TYPE_NAMES}

#: The Roles a Role comparison may pair (``CONTEXT.md``, *Role*).
ROLES: tuple[str, ...] = (opto_roles.ROLE_PAIRED, opto_roles.ROLE_YOKED,
                          opto_roles.ROLE_MUTUAL, opto_roles.ROLE_SELF,
                          opto_roles.ROLE_OPEN, opto_roles.ROLE_UNLIT)
_ROLE_BY_KEY = {r.lower(): r for r in ROLES}

#: The keys a definition may hold, in the order :func:`dump_definition` writes.
KEYS: tuple[str, ...] = ("format", "name", "display_name", "description",
                         "chamber_layout", "light", "required_wells", "factors",
                         "modules", "control", "plots", "plot_specs", "facets",
                         "constants")

#: A type's name: what ``experiment_type:`` holds, so a plain identifier.
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TypeDefinition:
    """One Experiment Type, as its yaml states it."""

    name: str
    display_name: str = ""
    description: str = ""
    #: ``single_well``, ``two_well``, or ``None`` for either.
    chamber_layout: str | None = None
    #: :data:`LIGHT_NONE`, :data:`LIGHT_ANY`, or a tuple of Paradigm type names.
    light: str | tuple[str, ...] = LIGHT_ANY
    required_wells: tuple[str, ...] = ()
    factors: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    #: The Role comparison's two Roles, or ``None``.
    roles: tuple[str, str] | None = None
    #: The Treatment control's factor, or ``None``.
    treatment_control: str | None = None
    plots: tuple[str, ...] = ()
    plot_specs: Mapping[str, Any] = field(default_factory=dict)
    facet_cutoffs: tuple[float, ...] | None = None
    facets_fixed: bool = False
    facet_labels: tuple[str, ...] = ()
    constants: Mapping[str, Any] = field(default_factory=dict)
    #: True for a type pyflic ships (read-only).
    shipped: bool = False
    #: The file it was read from, if any.
    source: Path | None = None

    @property
    def label(self) -> str:
        return self.display_name or self.name

    @property
    def expects_light(self) -> bool:
        """Whether a recording of this type is lit (``any`` may be)."""
        return self.light != LIGHT_NONE

    @property
    def paradigm_types(self) -> tuple[str, ...]:
        """The Paradigm types the type expects, or ``()`` for none and any."""
        return self.light if isinstance(self.light, tuple) else ()

    def light_text(self) -> str:
        """The expected light in words, for messages and the editor."""
        if self.light == LIGHT_NONE:
            return "no light"
        if self.light == LIGHT_ANY:
            return "any light or none"
        return ", ".join(self.light)

    def with_name(self, name: str, display_name: str | None = None) -> "TypeDefinition":
        """A copy named *name*, not shipped and from no file: *New from…*."""
        return replace(self, name=name, display_name=display_name or "",
                       shipped=False, source=None)


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def _strings(value: Any, where: str, problems: list[str]) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        problems.append(f"'{where}' must be a list")
        return ()
    out: list[str] = []
    for item in value:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def parse_light(value: Any) -> tuple[str | tuple[str, ...], list[str]]:
    """``(light, problems)`` for a ``light:`` value: ``none``, ``any``, or a
    list of Paradigm types in any case (read in PyMCU's spelling).  Unset is
    ``any``; ``false`` (yaml's ``no``) is ``none``."""
    if value is None or value is True:
        return LIGHT_ANY, []
    if value is False:
        return LIGHT_NONE, []
    if isinstance(value, str):
        key = value.strip().lower()
        if key in (LIGHT_NONE, "no", "off", ""):
            return LIGHT_NONE, []
        if key in (LIGHT_ANY, "yes"):
            return LIGHT_ANY, []
        value = [value]
    if not isinstance(value, (list, tuple)):
        return LIGHT_ANY, [f"'light' must be none, any or a list of Paradigm types, "
                           f"got {value!r}"]
    names: list[str] = []
    problems: list[str] = []
    for item in value:
        name = _PARADIGM_BY_KEY.get(str(item).strip().lower().replace(" ", ""))
        if name is None:
            problems.append(f"'light' names {item!r}, which is not a Paradigm type "
                            f"({', '.join(PARADIGM_TYPE_NAMES)})")
        elif name not in names:
            names.append(name)
    if not names:
        return (LIGHT_ANY if problems else LIGHT_NONE), problems
    return tuple(n for n in PARADIGM_TYPE_NAMES if n in names), problems


def _layout(value: Any, problems: list[str]) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if key in ("", LAYOUT_EITHER, "any"):
        return None
    if key not in DEFINITION_LAYOUTS:
        problems.append(f"'chamber_layout' must be single_well, two_well or either, "
                        f"got {value!r}")
        return None
    return key


def _roles(value: Any, problems: list[str]) -> tuple[str, str] | None:
    if value is None:
        return None
    names = _strings(value, "control.roles", problems)
    if not names:
        return None
    resolved = []
    for name in names:
        role = _ROLE_BY_KEY.get(name.lower())
        if role is None:
            problems.append(f"'control.roles' names {name!r}, which is not a Role "
                            f"({', '.join(ROLES)})")
        else:
            resolved.append(role)
    if len(names) != 2:
        problems.append(f"'control.roles' must name two Roles, got {len(names)}")
        return None
    if len(resolved) != 2:
        return None
    return resolved[0], resolved[1]


def _facets(value: Any, problems: list[str]) -> tuple[tuple[float, ...] | None, bool,
                                                       tuple[str, ...]]:
    if value is None:
        return None, False, ()
    if not isinstance(value, Mapping):
        problems.append("'facets' must be a mapping of cutoffs, fixed and labels")
        return None, False, ()
    cutoffs = None
    raw = value.get("cutoffs")
    if raw is not None:
        from .. import windowing

        try:
            cutoffs = tuple(windowing.normalize_cutoffs(raw))
        except (TypeError, ValueError) as err:
            problems.append(f"'facets.cutoffs': {err}")
    labels = _strings(value.get("labels"), "facets.labels", problems)
    if labels and cutoffs is not None and len(labels) != len(cutoffs) + 1:
        problems.append(f"'facets.labels' must name {len(cutoffs) + 1} windows, one more "
                        f"than the cutoffs, got {len(labels)}")
    fixed = bool(value.get("fixed", False))
    if fixed and cutoffs is None:
        problems.append("'facets.fixed' needs 'facets.cutoffs'")
        fixed = False
    for key in value:
        if key not in ("cutoffs", "fixed", "labels"):
            problems.append(f"'facets' has an unknown key {key!r}")
    return cutoffs, fixed, labels


def parse_definition(data: Any, *, source: Path | None = None,
                     shipped: bool = False) -> tuple[TypeDefinition | None, list[str]]:
    """``(definition, problems)`` from a definition's yaml mapping.

    Reading problems (an unknown key, a misspelt Paradigm type) are listed
    and the rest is still read, so the editor can show every one; the
    definition is ``None`` only when there is no usable ``name``.  Whether
    the modules can run together is :func:`definition_problems`'s question.
    """
    problems: list[str] = []
    if not isinstance(data, Mapping):
        return None, ["an Experiment Type definition must be a mapping"]
    fmt = data.get("format", FORMAT)
    try:
        if int(fmt) > FORMAT:
            problems.append(f"'format' {fmt} is newer than this pyflic reads ({FORMAT}); "
                            f"update pyflic")
    except (TypeError, ValueError):
        problems.append(f"'format' must be a whole number, got {fmt!r}")
    for key in data:
        if key not in KEYS:
            problems.append(f"unknown key {key!r} (known: {', '.join(KEYS)})")
    name = str(data.get("name") or "").strip()
    if not name:
        return None, problems + ["'name' is required"]
    if not _NAME_RE.match(name):
        problems.append(f"'name' {name!r} must start with a letter and hold only letters, "
                        f"digits and underscores, because experiment_type: names it")
    light, light_problems = parse_light(data.get("light"))
    problems += light_problems
    control = data.get("control") or {}
    if not isinstance(control, Mapping):
        problems.append("'control' must be a mapping of roles and treatment")
        control = {}
    for key in control:
        if key not in ("roles", "treatment"):
            problems.append(f"'control' has an unknown key {key!r}")
    treatment = str(control.get("treatment") or "").strip() or None
    cutoffs, fixed, labels = _facets(data.get("facets"), problems)
    constants = data.get("constants") or {}
    if not isinstance(constants, Mapping):
        problems.append("'constants' must be a mapping")
        constants = {}
    specs = data.get("plot_specs") or {}
    if not isinstance(specs, Mapping):
        problems.append("'plot_specs' must be a mapping of plot id to spec")
        specs = {}
    definition = TypeDefinition(
        name=name,
        display_name=str(data.get("display_name") or "").strip(),
        description=" ".join(str(data.get("description") or "").split()),
        chamber_layout=_layout(data.get("chamber_layout"), problems),
        light=light,
        required_wells=tuple(w.upper() for w in _strings(
            data.get("required_wells"), "required_wells", problems)),
        factors=_strings(data.get("factors"), "factors", problems),
        modules=tuple(m.lower() for m in _strings(data.get("modules"), "modules",
                                                  problems)),
        roles=_roles(control.get("roles"), problems),
        treatment_control=treatment,
        plots=_strings(data.get("plots"), "plots", problems),
        plot_specs={str(k): v for k, v in specs.items()},
        facet_cutoffs=cutoffs,
        facets_fixed=fixed,
        facet_labels=labels,
        constants=dict(constants),
        shipped=shipped,
        source=source,
    )
    return definition, problems


def read_definition(path: str | Path, *, shipped: bool = False) -> TypeDefinition:
    """The definition in *path*; ``ValueError`` naming the file and every
    problem, reading or structural, when it cannot be used."""
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as err:
        raise ValueError(f"{path}: cannot read the Experiment Type: {err}") from err
    definition, problems = parse_definition(data, source=path, shipped=shipped)
    if definition is not None:
        problems += definition_problems(definition)
    if problems or definition is None:
        raise ValueError(f"{path} is not a usable Experiment Type:\n  - "
                         + "\n  - ".join(problems))
    return definition


# ---------------------------------------------------------------------------
# Whether the modules can run together
# ---------------------------------------------------------------------------

def definition_problems(definition: TypeDefinition) -> list[str]:
    """What stops *definition* from running, beyond reading it: a module it
    names that pyflic lacks, a module whose needs the type does not meet, two
    modules that exclude each other, a control comparison it cannot make.
    Never raises."""
    from ..analysis_modules import get_module, module_ids

    problems: list[str] = []
    known = set(module_ids())
    modules = []
    for module_id in definition.modules:
        if module_id not in known:
            problems.append(f"module {module_id!r} is not an Analysis Module "
                            f"({', '.join(sorted(known))})")
        else:
            modules.append(get_module(module_id))
    for module in modules:
        problems += module.needs_problems(definition)
        for other in modules:
            if other.id in module.conflicts and module.id < other.id:
                problems.append(f"modules {module.id!r} and {other.id!r} cannot run "
                                f"together: {module.conflict_reason}")
    classes = [m for m in modules if m.experiment_class]
    if len(classes) > 1:
        problems.append(f"modules {', '.join(repr(m.id) for m in classes)} each bring an "
                        f"analysis class of their own; a type may include one of them")
    if definition.roles is not None:
        a, b = definition.roles
        if a == b:
            problems.append(f"'control.roles' compares {a} with itself")
        if not definition.expects_light:
            problems.append("'control.roles' needs light: a Role comes from the Opto "
                            "Program, and the type expects no light")
        if {a, b} == {"Paired", "Yoked"} \
                and not {"yoked", "ratio"} & set(definition.modules):
            problems.append("'control.roles' Paired against Yoked is the paired − yoked "
                            "comparison: include the 'yoked' module (or 'ratio', whose "
                            "Paired-Yoked Difference it is)")
    if definition.treatment_control is not None \
            and definition.treatment_control not in definition.factors:
        problems.append(f"'control.treatment' names {definition.treatment_control!r}, "
                        f"which is not one of the type's factors "
                        f"({', '.join(definition.factors) or 'none'})")
    layout = definition.chamber_layout
    if layout == "single_well" and definition.required_wells \
            and set(definition.required_wells) - {"A"}:
        problems.append("a single-well type can require only well A")
    from ..opto_qc import opto_constant_problems

    problems += opto_constant_problems(definition.constants)
    from ..ceiling import ceiling_problems

    problems += ceiling_problems(definition.constants)
    for module in modules:
        problems += module.constant_problems(definition.constants)
    return problems


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def dump_definition(definition: TypeDefinition) -> dict:
    """*definition* as the yaml mapping :func:`parse_definition` reads,
    leaving out what is unset, in :data:`KEYS` order."""
    d = definition
    out: dict[str, Any] = {"format": FORMAT, "name": d.name}
    if d.display_name:
        out["display_name"] = d.display_name
    if d.description:
        out["description"] = d.description
    out["chamber_layout"] = d.chamber_layout or LAYOUT_EITHER
    out["light"] = list(d.light) if isinstance(d.light, tuple) else d.light
    if d.required_wells:
        out["required_wells"] = list(d.required_wells)
    if d.factors:
        out["factors"] = list(d.factors)
    if d.modules:
        out["modules"] = list(d.modules)
    control: dict[str, Any] = {}
    if d.roles is not None:
        control["roles"] = list(d.roles)
    if d.treatment_control:
        control["treatment"] = d.treatment_control
    if control:
        out["control"] = control
    if d.plots:
        out["plots"] = list(d.plots)
    if d.plot_specs:
        out["plot_specs"] = {k: v for k, v in d.plot_specs.items()}
    facets: dict[str, Any] = {}
    if d.facet_cutoffs is not None:
        facets["cutoffs"] = [_clean(v) for v in d.facet_cutoffs]
        if d.facets_fixed:
            facets["fixed"] = True
    if d.facet_labels:
        facets["labels"] = list(d.facet_labels)
    if facets:
        out["facets"] = facets
    if d.constants:
        out["constants"] = dict(d.constants)
    return out


def write_definition(definition: TypeDefinition, path: str | Path) -> Path:
    """Write *definition* to *path* as yaml; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dump_definition(definition), sort_keys=False,
                          allow_unicode=True, default_flow_style=False)
    path.write_text(text, encoding="utf-8")
    return path


def same_definition(a: TypeDefinition, b: TypeDefinition) -> bool:
    """Whether *a* and *b* say the same thing, wherever they were read from."""
    return dump_definition(a) == dump_definition(b)


def _clean(value):
    try:
        return int(value) if float(value) == int(value) else float(value)
    except (TypeError, ValueError):
        return value

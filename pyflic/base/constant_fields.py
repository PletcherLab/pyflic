"""Design constants described once: how the editors show them, what is valid.

A design constant (a key under ``global.constants``) is read by the analysis,
shown by the Project Design dialog and the Config Editor, and checked by the
type's validation, ``pyflic lint`` and the loader.  Describing each one here,
once, is what stops those five from disagreeing about what exists or what is
allowed.  Progressive Ratio's constants (``experiment_types.progressive_ratio``)
and the optogenetic light QC's (``opto_qc``) are both lists of these.

Imports nothing from pyflic, so any module may use it without a cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ConstantField:
    """One design constant: how the Project Design dialog and the Config
    Editor show it, and what is accepted for it.

    *group* is ``"switch"`` (true / false) or a section name the editors lay
    out by; *short_label* and *choices* are the Config Editor's words for a
    switch, whose row label must stay short and whose picker names what each
    value does.  *blank* says what an unset value means when there is no
    default for it.
    """

    key: str
    label: str
    tooltip: str
    group: str
    integer: bool = False
    minimum: float | None = None
    maximum: float | None = None
    minimum_exclusive: bool = False
    blank: str | None = None
    short_label: str | None = None
    choices: tuple[str, str] = ("yes", "no")


def constant_problems(fields: Iterable[ConstantField],
                      constants: Mapping[str, Any] | None, *,
                      default: str = "the default") -> list[str]:
    """What is wrong with the *fields* a ``constants:`` block states — a switch
    that is not true or false, a threshold that is not a number or falls
    outside its range.  Keys it does not state are fine: defaults fill them
    in.  *default* names what an emptied key falls back to in the message.
    Never raises."""
    problems: list[str] = []
    stated = dict(constants or {})
    for field in fields:
        if field.key not in stated:
            continue
        value = stated[field.key]
        where = f"'constants.{field.key}'"
        if value is None:
            if field.blank is None:
                problems.append(f"{where} is empty; give a value or remove the key "
                                f"to use {default}")
            continue
        if field.group == "switch":
            if not isinstance(value, bool):
                problems.append(f"{where} must be true or false, got {value!r}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            problems.append(f"{where} must be a number, got {value!r}")
            continue
        v = float(value)
        if field.integer and v != int(v):
            problems.append(f"{where} must be a whole number, got {value!r}")
        if field.minimum is not None and (
                v < field.minimum or (field.minimum_exclusive and v == field.minimum)):
            bound = "greater than" if field.minimum_exclusive else "at least"
            problems.append(f"{where} must be {bound} {field.minimum:g}, got {value!r}")
        if field.maximum is not None and v > field.maximum:
            problems.append(f"{where} must be at most {field.maximum:g}, got {value!r}")
    return problems

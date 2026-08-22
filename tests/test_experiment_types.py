"""Tests for the Experiment Type / Chamber Layout split (ADR-0007)."""

from __future__ import annotations

import pytest

from pyflic.base import experiment_types as et
from pyflic.base import windowing


def test_custom_is_the_absence_of_a_type():
    assert et.get_experiment_type(None).name == "Custom"
    assert et.get_experiment_type("").name == "Custom"
    assert et.get_experiment_type(None).is_custom


def test_registry_lists_the_shipped_types():
    names = [t.name for t in et.available_experiment_types()]
    assert names[0] == "Custom"          # Custom always first
    assert set(names) == {"Custom", "Hedonic", "ProgressiveRatio"}


@pytest.mark.parametrize("retired,layout", [
    ("two_well", "two_well"),
    ("single_well", "single_well"),
    ("Two-Well", "two_well"),
])
def test_retired_layout_names_raise_with_the_migration(retired, layout):
    with pytest.raises(ValueError, match=rf"chamber_layout: {layout}"):
        et.get_experiment_type(retired)


def test_a_type_owns_its_chamber_layout():
    hedonic = et.get_experiment_type("Hedonic")
    assert hedonic.chamber_layout == "two_well"
    # The yaml gets no vote: even a contradicting value is ignored here and
    # reported by validate().
    assert hedonic.resolve_chamber_layout({"chamber_layout": "single_well"}) \
        == "two_well"
    assert hedonic.resolve_chamber_size({}) == 2


def test_custom_reads_its_layout_from_the_config():
    custom = et.get_experiment_type("Custom")
    assert custom.resolve_chamber_layout({"chamber_layout": "single_well"}) \
        == "single_well"
    assert custom.resolve_chamber_size({"chamber_layout": "single_well"}) == 1
    # Absent -> the historical default.
    assert custom.resolve_chamber_layout({}) == "two_well"


def test_unknown_layout_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown chamber_layout"):
        et.get_experiment_type("Custom").resolve_chamber_layout(
            {"chamber_layout": "three_well"})


def test_validate_flags_keys_the_type_owns():
    problems = et.get_experiment_type("Hedonic").validate({
        "chamber_layout": "two_well",
        "params": {"chamber_size": 2},
        "well_names": {"A": "S5", "B": "S5Y5"},
    })
    assert any("chamber_layout" in p for p in problems)
    assert any("chamber_size" in p for p in problems)


def test_a_two_well_type_requires_both_wells_named():
    """A PI axis reading 'preference for A' tells a reader nothing."""
    problems = et.get_experiment_type("Hedonic").validate({
        "well_names": {"A": "S5"}})
    assert any("requires well_names for B" in p for p in problems)


def test_phase_labels_apply_only_at_the_types_default_cutoffs():
    pr = et.get_experiment_type("ProgressiveRatio")
    default = windowing.facet_windows(pr.resolve_facet_cutoffs({}))
    assert pr.phase_labels_for(default, {}) == ["Training", "Test"]
    # Move the cutoff and the named phases stop being true.
    moved = windowing.facet_windows([15])
    assert pr.phase_labels_for(moved, {"facet_cutoffs": [15]}) == \
        ["0-15 min", "15+ min"]


def test_explicit_facet_labels_win():
    pr = et.get_experiment_type("ProgressiveRatio")
    windows = windowing.facet_windows([30])
    assert pr.phase_labels_for(windows, {"facet_labels": ["Early", "Late"]}) \
        == ["Early", "Late"]


def test_yaml_constants_layer_over_the_type_defaults():
    hedonic = et.get_experiment_type("Hedonic")
    merged = hedonic.resolve_constants({"constants": {"max_events_cutoff": 5.0}})
    assert merged["max_events_cutoff"] == 5.0
    assert merged["min_untransformed_licks_cutoff"] == 20   # type default kept


def test_build_global_omits_what_the_type_owns():
    built = et.get_experiment_type("Hedonic").build_global(
        params={"feeding_threshold": 20, "chamber_size": 2},
        well_names={"A": "S5", "B": "S5Y5"})
    assert built["experiment_type"] == "Hedonic"
    assert "chamber_layout" not in built
    assert "chamber_size" not in built["params"]


def test_report_set_depends_on_the_chamber_layout():
    custom = et.get_experiment_type("Custom")
    assert "faceted_pi" in custom.report_set("two_well")
    # A preference index is meaningless with one well.
    assert "faceted_pi" not in custom.report_set("single_well")


def test_primary_phase_is_the_second_window_when_there_are_several():
    base = et.get_experiment_type("Custom")
    assert base.primary_phase_index(windowing.facet_windows([10, 70])) == 1
    assert base.primary_phase_index([(0, float("inf"))]) == 0

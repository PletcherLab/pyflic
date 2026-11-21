"""A recording's description travels with its config.

``about:`` (ADR-0024) says what a recording is, for a reader, and pyflic never
reads it for an analysis.  The Design Copy (ADR-0025) puts the Design's
description of the recording — type, layout, well names, factors, control
level — in every Member's ``global:``, so each config reads on its own, and
writing it never overwrites what a Member says its recording was.

Both are optional: a config or a Member without them behaves as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pyflic.base import about as about_mod
from pyflic.base import cache
from pyflic.base import project as project_mod
from pyflic.base.project import Project
from pyflic.base.yaml_lint import lint_flic_config

DESIGN = {
    "experiment_type": "Hedonic",
    "transform_licks": False,
    "params": {"feeding_threshold": 20, "samples_per_second": 5},
    "well_names": {"A": "S5", "B": "S5Y5"},
    "experimental_design_factors": {"Genotype": ["Ctrl", "MBON10", "MBON11"],
                                    "Sex": ["M", "F"]},
}
COPY = {"experiment_type": "Hedonic",
        "well_names": {"A": "S5", "B": "S5Y5"},
        "experimental_design_factors": DESIGN["experimental_design_factors"]}
DFMS = [{"id": 1, "chambers": {1: "Ctrl, M", 2: "MBON10, F"}}]


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _project(tmp_path: Path, members: dict[str, dict], design=DESIGN) -> Path:
    root = tmp_path / "Proj"
    _write(root / "project.yaml", {"name": "Proj", "design": {"global": design}})
    for name, config in members.items():
        _write(root / name / "flic_config.yaml", config)
        (root / name / "data").mkdir(parents=True, exist_ok=True)
        (root / name / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    return root


# ---------------------------------------------------------------------------
# about:
# ---------------------------------------------------------------------------

def test_clean_about_drops_blanks_orders_keys_and_keeps_unknown_ones():
    cleaned = about_mod.clean_about({
        "notes": "  ", "lab_book": "p. 12", "title": " PR rep 1 ",
        "sex": ["female", ""], "protocol": {"name": "FLIC PR", "version": ""},
        "factors": {"Genotype": {"kind": "Genotype", "levels": {"Ctrl": " ", "X": "a > b"}},
                    "Empty": {"kind": "", "levels": {}}},
    })
    assert list(cleaned) == ["title", "sex", "protocol", "factors", "lab_book"]
    assert cleaned["title"] == "PR rep 1"
    assert cleaned["sex"] == ["female"]
    assert cleaned["protocol"] == {"name": "FLIC PR"}
    assert cleaned["factors"] == {"Genotype": {"kind": "genotype", "levels": {"X": "a > b"}}}
    assert about_mod.clean_about({"title": "", "factors": {}}) == {}


def test_about_problems_are_warnings_about_kinds_and_undeclared_names():
    problems = dict(about_mod.about_problems(
        {"title": "x" * 61,
         "factors": {"Genotype": {"kind": "strain", "levels": {"Ctrl": "w", "Nope": "?"}},
                     "Dose": {"kind": "treatment"}}},
        {"Genotype": ["Ctrl", "MBON10"]}))
    assert "61 characters" in problems["about.title"]
    assert "strain" in problems["about.factors.Genotype.kind"]
    assert "Nope" in problems["about.factors.Genotype.levels.Nope"]
    assert "does not declare" in problems["about.factors.Dose"]
    assert about_mod.about_problems(None) == []
    assert about_mod.about_problems("text")[0][0] == "about"


def test_a_level_meaning_may_be_text_or_a_mapping():
    notes = about_mod.factor_notes({"factors": {"G": {"levels": {
        "a": "one", "b": {"definition": "two", "term": "GEN-0001"}}}}})
    assert notes["G"]["levels"] == {"a": "one", "b": "two"}


# ---------------------------------------------------------------------------
# The linter and the cache: about: is accepted and changes no number
# ---------------------------------------------------------------------------

def test_the_linter_accepts_about_and_only_warns_on_it(tmp_path):
    config = tmp_path / "exp" / "flic_config.yaml"
    _write(config, {"about": {"title": "Rep 1",
                              "factors": {"Genotype": {"kind": "strain"}}},
                    "global": {"experimental_design_factors": {"Genotype": ["Ctrl"]}},
                    "dfms": [{"id": 1, "chambers": {1: "Ctrl"}}]})
    issues = lint_flic_config(config)
    assert not any("unknown top-level key" in i.message for i in issues)
    about_issues = [i for i in issues if (i.path or "").startswith("about")]
    assert [i.severity for i in about_issues] == ["warning"]
    assert "strain" in about_issues[0].message


def test_editing_about_keeps_the_cache_key_and_no_about_keeps_the_old_one(tmp_path):
    exp = tmp_path / "exp"
    plain = {"global": {"params": {"feeding_threshold": 20}}, "dfms": DFMS}
    _write(exp / "flic_config.yaml", plain)
    import hashlib
    raw = (exp / "flic_config.yaml").read_bytes()
    ## A config without about: hashes byte for byte, as it always has.
    assert cache._config_hash(exp) == hashlib.sha256(raw).hexdigest()[:16]

    _write(exp / "flic_config.yaml", {"about": {"title": "one"}, **plain})
    first = cache._config_hash(exp)
    _write(exp / "flic_config.yaml", {"about": {"title": "two", "notes": "x"}, **plain})
    assert cache._config_hash(exp) == first
    _write(exp / "flic_config.yaml",
           {"about": {"title": "two"}, "global": {"params": {"feeding_threshold": 21}},
            "dfms": DFMS})
    assert cache._config_hash(exp) != first


# ---------------------------------------------------------------------------
# The Design Copy
# ---------------------------------------------------------------------------

def test_the_design_copy_holds_only_what_the_recording_is():
    assert project_mod.design_copy(DESIGN) == COPY
    assert project_mod.design_copy({"params": {"x": 1}}) == {}


def test_a_member_carrying_the_copy_loads_and_one_without_still_does(tmp_path):
    root = _project(tmp_path, {"rep1": {"global": COPY, "dfms": DFMS},
                               "rep2": {"dfms": DFMS}})
    project = Project(root)
    assert project.member_names == ["rep1", "rep2"]
    assert project.resolved_global("rep1")["params"]["feeding_threshold"] == 20
    ## The one without a copy is named, never refused.
    assert any("rep2 carries no copy" in w for w in project.warnings)
    assert not any("rep1" in w and "copy" in w for w in project.warnings)


def test_a_copy_that_differs_from_the_design_is_a_load_error(tmp_path):
    stale = {**COPY, "well_names": {"A": "S5", "B": "Yeast"}}
    root = _project(tmp_path, {"rep1": {"global": stale, "dfms": DFMS}})
    with pytest.raises(ValueError, match="well_names"):
        Project(root)


@pytest.mark.parametrize("own, words", [
    ({"well_names": {"B": "Yeast"}}, "well B holds 'Yeast'"),
    ({"experiment_type": "ProgressiveRatio"}, "experiment_type is 'ProgressiveRatio'"),
    ({"chamber_layout": "single_well"}, "single_well"),
    ({"experimental_design_factors": {"Sex": ["M"], "Genotype": ["Ctrl"]}}, "in that order"),
    ({"experimental_design_factors": {"Genotype": ["Ctrl", "MBON99"], "Sex": ["M"]}},
     "MBON99"),
    ({"control_level": "Ctrl"}, None),            # the design names none: filled in
])
def test_description_conflicts(own, words):
    problems = project_mod.description_conflicts({"global": own, "dfms": DFMS}, DESIGN)
    if words is None:
        assert problems == []
    else:
        assert any(words in p for p in problems), problems


def test_fewer_levels_and_a_custom_member_are_not_conflicts():
    ## A genotype panel: each member declares the levels it used.
    fewer = {"experimental_design_factors": {"Genotype": ["Ctrl", "MBON10"],
                                             "Sex": ["M", "F"]},
             "params": {"feeding_threshold": 99}}
    assert project_mod.description_conflicts({"global": fewer, "dfms": DFMS}, DESIGN) == []
    ## Custom says "unspecified": the Design's type is taken when the layout agrees.
    custom = {"chamber_layout": "two_well", "well_names": {"A": "S5"}}
    assert project_mod.description_conflicts({"global": custom, "dfms": DFMS}, DESIGN) == []


def test_chambers_that_do_not_fit_the_designs_factors_are_a_conflict():
    plain = {"global": {"well_names": {"A": "S5"}},
             "dfms": [{"id": 1, "chambers": {1: "Sucrose"}}]}
    assert any("not one level each" in p
               for p in project_mod.description_conflicts(plain, DESIGN))
    ## A Member with no global: of its own was always read against the design.
    assert project_mod.description_conflicts({"dfms": DFMS}, DESIGN) == []


def test_adopting_writes_the_copy_and_leaves_a_conflicting_member_untouched(tmp_path):
    conflicting = {"about": {"title": "rep3"},
                   "global": {**COPY, "well_names": {"A": "S5", "B": "Yeast"},
                              "params": {"feeding_threshold": 5}},
                   "dfms": DFMS}
    root = _project(tmp_path, {
        "rep1": {"dfms": DFMS, "scripts": [{"name": "mine", "steps": []}]},
        "rep2": {"global": {"experimental_design_factors": {"Genotype": ["Ctrl"],
                                                             "Sex": ["M", "F"]},
                            "params": {"feeding_threshold": 99}},
                 "dfms": [{"id": 1, "chambers": {1: "Ctrl, M"}}]},
        "rep3": conflicting,
        "rep4": {"global": COPY, "dfms": DFMS},
    })
    states = {s.name: s for s in project_mod.member_design_states(root)}
    assert states["rep1"].state == "write"
    assert states["rep2"].state == "write"
    assert states["rep3"].state == "conflict"
    assert "well B holds 'Yeast'" in states["rep3"].reasons[0]
    assert states["rep4"].state == "in_step"
    before = (root / "rep3" / "flic_config.yaml").read_bytes()

    assert project_mod.adopt_design(root) == ["rep1", "rep2"]

    rep1 = yaml.safe_load((root / "rep1" / "flic_config.yaml").read_text())
    assert rep1["global"] == COPY
    assert rep1["scripts"] == [{"name": "mine", "steps": []}]
    assert list(rep1) == ["global", "dfms", "scripts"]
    rep2 = yaml.safe_load((root / "rep2" / "flic_config.yaml").read_text())
    ## Widened to the design's levels; its chambers, which say which flies had
    ## which, are untouched.  The analysis setting is the design's.
    assert rep2["global"] == COPY
    assert rep2["dfms"] == [{"id": 1, "chambers": {1: "Ctrl, M"}}]
    ## Never rewritten: its config says what its recording was.
    assert (root / "rep3" / "flic_config.yaml").read_bytes() == before
    with pytest.raises(ValueError, match="well_names"):
        Project(root)


def test_member_global_is_none_for_a_conflict():
    assert project_mod.member_global({"global": COPY, "dfms": DFMS}, DESIGN) == COPY
    clash = {"global": {"well_names": {"A": "Water"}}, "dfms": DFMS}
    assert project_mod.member_global(clash, DESIGN) is None


def test_scaffolding_writes_the_copy_but_not_another_members_about(tmp_path):
    root = _project(tmp_path, {"rep1": {"about": {"title": "rep1"}, "global": COPY,
                                        "dfms": DFMS}})
    (root / "rep2" / "data").mkdir(parents=True)
    (root / "rep2" / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    config, _notes = Project(root).scaffold_member_config("rep2")
    assert config["global"] == COPY
    assert "about" not in config


# ---------------------------------------------------------------------------
# A first Design from a folder of experiments
# ---------------------------------------------------------------------------

def test_a_first_design_declares_every_level_any_member_used(tmp_path):
    root = tmp_path / "Panel"
    _write(root / "a" / "flic_config.yaml",
           {"global": {"experiment_type": "Hedonic", "params": {"feeding_threshold": 20},
                       "experimental_design_factors": {"Genotype": ["Ctrl", "MBON10"]}},
            "dfms": DFMS})
    _write(root / "b" / "flic_config.yaml",
           {"global": {"experiment_type": "Hedonic",
                       "experimental_design_factors": {"Genotype": ["Ctrl", "MBON11"]}},
            "dfms": DFMS})
    design = project_mod.design_from_members(root)
    assert design["experimental_design_factors"] == {"Genotype": ["Ctrl", "MBON10", "MBON11"]}
    assert design["params"] == {"feeding_threshold": 20}      # the first member's


def test_members_with_other_factors_are_not_merged(tmp_path):
    root = tmp_path / "Mixed"
    _write(root / "a" / "flic_config.yaml",
           {"global": {"experimental_design_factors": {"Genotype": ["Ctrl"]}}, "dfms": []})
    _write(root / "b" / "flic_config.yaml",
           {"global": {"experimental_design_factors": {"Dose": ["1"]}}, "dfms": []})
    assert project_mod.design_from_members(root)["experimental_design_factors"] == \
        {"Genotype": ["Ctrl"]}


# ---------------------------------------------------------------------------
# about: across a Project: never an error, sometimes worth a warning
# ---------------------------------------------------------------------------

def test_members_describing_one_level_two_ways_are_warned_of(tmp_path):
    def member(meaning, kind="genotype"):
        return {"about": {"title": "anything",
                          "factors": {"Genotype": {"kind": kind,
                                                   "levels": {"MBON10": meaning}}}},
                "global": COPY, "dfms": DFMS}
    root = _project(tmp_path, {"rep1": member("R12C11 > GtACR1"),
                               "rep2": member("R12C11 > Kir2.1", kind="treatment"),
                               "rep3": {"global": COPY, "dfms": DFMS}})
    project = Project(root)                       # loads: about: is never an error
    spread = [w for w in project.warnings if w.startswith("about:")]
    assert len(spread) == 2
    assert any("different kinds" in w for w in spread)
    assert any("'MBON10' differently" in w and "rep1" in w and "rep2" in w for w in spread)


def test_members_that_agree_or_say_nothing_raise_no_about_warning(tmp_path):
    same = {"about": {"title": "x", "factors": {"Genotype": {"levels": {"Ctrl": "w1118"}}}},
            "global": COPY, "dfms": DFMS}
    root = _project(tmp_path, {"rep1": same, "rep2": dict(same, about={"title": "y"}),
                               "rep3": {"global": COPY, "dfms": DFMS}})
    assert not [w for w in Project(root).warnings if w.startswith("about:")]


def test_a_member_that_states_no_layout_takes_the_designs():
    single = {"chamber_layout": "single_well", "params": {}}
    ## Says nothing about the layout: inherits it, no conflict.
    assert project_mod.description_conflicts(
        {"global": {"transform_licks": False}, "dfms": []}, single) == []
    assert project_mod.description_conflicts({"dfms": []}, single) == []
    ## A legacy chamber_size is a statement, and this one disagrees.
    assert any("two_well" in p for p in project_mod.description_conflicts(
        {"global": {"params": {"chamber_size": 2}}, "dfms": []}, single))


# ---------------------------------------------------------------------------
# Scaffolding: a new Member starts from the first one's shared About (ADR-0026)
# ---------------------------------------------------------------------------
#
# Replicates share who ran them, the series, the flies, the protocol and what
# each factor and level means, so a scaffolded Member takes those from the
# first Member.  The title, the purpose, the notes and run: are one
# recording's alone, and are never copied.

FILED = {
    "run": "R-20261121-scott-01",
    "title": "Hedonic panel, replicate 1",
    "purpose": "Whether silencing MBON10 changes hedonic feeding.",
    "experimenter": "scott",
    "series": "hedonic-2026",
    "organism": "Drosophila melanogaster",
    "sex": ["female"],
    "age_at_start": "5-7 days",
    "protocol": {"name": "FLIC hedonic", "version": "2"},
    "notes": "Lid 3 was replaced.",
    "factors": {
        "Genotype": {"kind": "genotype", "levels": {
            "Ctrl": {"definition": "w1118 > UAS-GtACR1", "term": "GEN-0012",
                     "label": "w1118 x UAS-GtACR1", "role": "reference"},
            "MBON10": "R12C11-GAL4 > UAS-GtACR1"}},
        "Sex": {"kind": "sex"},
    },
    "x_lab": {"a": 1},
}


def _recording(root: Path, name: str) -> None:
    (root / name / "data").mkdir(parents=True)
    (root / name / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")


def test_a_scaffolded_member_holds_the_first_members_shared_about(tmp_path):
    root = _project(tmp_path, {"rep1": {"about": FILED, "global": COPY, "dfms": DFMS}})
    _recording(root, "rep2")
    path, notes = Project(root).scaffold_member("rep2")
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    assert list(config) == ["about", "global", "dfms"]
    assert config["about"] == {key: FILED[key] for key in about_mod.SHARED_KEYS}
    for key in ("run", "title", "purpose", "notes", "x_lab"):
        assert key not in config["about"]
    ## Each level's whole entry: what a catalogue standardised once travels.
    assert config["about"]["factors"]["Genotype"]["levels"]["Ctrl"]["term"] == "GEN-0012"
    assert any(note.startswith("about:") and "rep1" in note for note in notes)
    ## The first Member's own file is only read.
    assert yaml.safe_load((root / "rep1" / "flic_config.yaml").read_text())["about"] == FILED
    project = Project(root)
    assert project.member_names == ["rep1", "rep2"]
    assert not [w for w in project.warnings if w.startswith("about:")]


def test_scaffolding_reads_the_first_members_about_as_it_is_on_disk(tmp_path):
    """The Hub keeps one Project open while a catalogue files its Members."""
    typed = {"about": {"title": "rep1", "series": "typed"}, "global": COPY, "dfms": DFMS}
    root = _project(tmp_path, {"rep1": typed})
    _recording(root, "rep2")
    project = Project(root)
    _write(root / "rep1" / "flic_config.yaml", {**typed, "about": FILED})
    config, _notes = project.scaffold_member_config("rep2")
    assert config["about"]["series"] == "hedonic-2026"
    assert config["about"]["factors"]["Genotype"]["levels"]["Ctrl"]["term"] == "GEN-0012"
    ## A file that is gone leaves what the Project read when it opened.
    (root / "rep1" / "flic_config.yaml").unlink()
    config, _notes = project.scaffold_member_config("rep2")
    assert config["about"] == {"series": "typed"}


def test_a_scaffold_takes_only_what_the_design_declares_of_the_factors(tmp_path):
    about = {"series": "s", "factors": {
        "Genotype": {"kind": "genotype",
                     "levels": {"Ctrl": "w1118", "Gone": "a level the design dropped"}},
        "Dose": {"kind": "treatment", "levels": {"1": "1 mM"}}}}
    root = _project(tmp_path, {"rep1": {"about": about, "global": COPY, "dfms": DFMS}})
    _recording(root, "rep2")
    config, _notes = Project(root).scaffold_member_config("rep2")
    assert config["about"] == {"series": "s", "factors": {
        "Genotype": {"kind": "genotype", "levels": {"Ctrl": "w1118"}}}}


def test_a_first_member_that_shares_nothing_leaves_a_scaffold_without_about(tmp_path):
    only_its_own = {"run": "R-20261121-scott-01", "title": "rep1", "purpose": "p",
                    "notes": "n", "x_lab": {"a": 1}}
    root = _project(tmp_path, {"rep1": {"about": only_its_own, "global": COPY,
                                        "dfms": DFMS}})
    _recording(root, "rep2")
    config, notes = Project(root).scaffold_member_config("rep2")
    assert "about" not in config
    assert not any(note.startswith("about:") for note in notes)

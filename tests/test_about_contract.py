"""``about:`` as a contract shared with a catalogue (ADR-0024, ADR-0026).

A catalogue files a recording from its folder and afterwards rewrites the
config's ``about:`` to match its own record: it adds ``run:`` and, inside a
level's mapping, ``term:``, ``label:`` and ``role:``, possibly while a pyflic
window has the file open.  So the section must be:

* **inert**: no number, cached summary or "re-run needed" depends on it;
* **kept**: every writer of ``flic_config.yaml`` leaves it equal in value;
* **fresh**: a writer that does not edit it writes back the section on disk
  when it saves, not the one it read when it opened the file;
* **never an error**: anything odd in it is a warning, and the config still
  loads, saves and runs.

The writers of a ``flic_config.yaml`` in pyflic, each of which is exercised
here or in the file named:

==========================================  ==================================
The Experiment Editor's save                edits ``about:``
                                            (``test_experiment_editor_ui.py``)
The Config Editor's save                    does not; reads it again at save
                                            (``test_config_editor_ui.py``)
The Script Editor's save                    does not; re-reads the whole file
A Selection tick (``selections``)           does not; re-reads the whole file
``project.adopt_design`` (saving a Design)  does not; re-reads the whole file
``Project.scaffold_member``                 writes a new file: the shared About
The Hub's *Copy config from…*               writes a new file: the shared About
                                            (``test_project_creation_ui.py``)
==========================================  ==================================
"""

from __future__ import annotations

import contextlib
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from pyflic.base import about as about_mod
from pyflic.base import cache
from pyflic.base import project as project_mod
from pyflic.base import selections as sel
from pyflic.base.project import Project
from pyflic.base.yaml_config import load_experiment_yaml
from pyflic.base.yaml_lint import lint_flic_config

#: Every field, an unknown key, ``run:``, and a level mapping holding what a
#: catalogue writes and a key nobody knows.
ABOUT = {
    "run": "R-20261121-scott-01",
    "title": "Two-choice, sucrose against yeast, replicate 1",
    "purpose": "Whether silencing MBON10 changes what flies choose.",
    "experimenter": "scott",
    "series": "two-choice-2026",
    "organism": "Drosophila melanogaster",
    "sex": ["female"],
    "age_at_start": "5-7 days",
    "protocol": {"name": "FLIC two-choice", "version": "2", "link": "https://example.org/tc"},
    "notes": "Lid 3 was replaced.",
    "factors": {
        "Genotype": {
            "kind": "genotype",
            "levels": {
                "Exp": "R12C11-GAL4 > UAS-GtACR1",
                "Ctrl": {"definition": "w1118 > UAS-GtACR1", "term": "GEN-0012",
                         "label": "w1118 x UAS-GtACR1", "role": "reference",
                         "x_other": "kept"},
            },
        },
    },
    "x_lab": {"a": 1},
}
FACTORS = {"Genotype": ["Ctrl", "Exp"]}
PARAMS = {"feeding_threshold": 5, "feeding_minimum": 5, "tasting_minimum": 1,
          "tasting_maximum": 4, "feeding_event_link_gap": 2, "samples_per_second": 5,
          "correct_for_dual_feeding": False}
CHAMBERS = {1: "Ctrl", 2: "Ctrl", 3: "Exp", 4: "Exp", 5: "Ctrl", 6: "Exp"}
DESIGN = {"transform_licks": False, "params": PARAMS,
          "experimental_design_factors": FACTORS}
DFMS = [{"id": 1, "chambers": CHAMBERS}]


def _edited(about: dict) -> dict:
    """*about* after a person retitles it and rewords a level, and a catalogue
    files it again under another id and another term."""
    out = copy.deepcopy(about)
    out["run"] = "R-20261122-scott-07"
    out["title"] = "Retitled"
    out["factors"]["Genotype"]["levels"]["Exp"] = "another wording"
    out["factors"]["Genotype"]["levels"]["Ctrl"]["definition"] = "the control cross"
    out["factors"]["Genotype"]["levels"]["Ctrl"]["term"] = "GEN-0099"
    return out


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _standalone(folder: Path, about=ABOUT, **extra) -> Path:
    """A standalone recording's config; returns its path."""
    config = {"global": dict(DESIGN), "dfms": copy.deepcopy(DFMS), **extra}
    if about is not None:
        config = {"about": copy.deepcopy(about), **config}
    _write(folder / "flic_config.yaml", config)
    return folder / "flic_config.yaml"


def _member(root: Path, name: str, about=ABOUT, **extra) -> Path:
    """A Member's config under the Project at *root*; returns its path."""
    if not (root / "project.yaml").exists():
        _write(root / "project.yaml", {"name": root.name, "design": {"global": DESIGN}})
    config = {"dfms": copy.deepcopy(DFMS), **extra}
    if about is not None:
        config = {"about": copy.deepcopy(about), **config}
    _write(root / name / "flic_config.yaml", config)
    return root / name / "flic_config.yaml"


def _write_csv(directory: Path, *, minutes: float = 10.0) -> None:
    """Five samples a second with a feeding bout on every well every minute."""
    n = int(minutes * 60 * 5)
    rng = np.random.default_rng(1)
    cols: dict[str, object] = {"Sample": np.arange(1, n + 1),
                               "Seconds": np.arange(n) / 5.0}
    for w in range(1, 13):
        sig = rng.normal(0.0, 0.3, size=n)
        for k in range(int(minutes)):
            start = 300 * k + 40 + (w * 7) % 30
            sig[start:start + 8] += 10.0 + rng.normal(0, 0.5, size=8)
        cols[f"W{w}"] = np.round(sig, 3)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_csv(directory / "DFM1_0.csv", index=False)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        result = fn(*args, **kwargs)
    return result, out.getvalue()


# ---------------------------------------------------------------------------
# Inert: no number, cached summary or "re-run needed" depends on it
# ---------------------------------------------------------------------------

def _tables(member: Path) -> dict[str, bytes]:
    """Every summary table an analysis wrote, by file name."""
    return {path.name: path.read_bytes()
            for path in sorted((member / "analysis").glob("*.csv"))}


def test_an_analysis_is_the_same_with_no_about_a_full_one_and_an_edited_one(tmp_path):
    root = tmp_path / "Proj"
    plain = _member(root, "plain", about=None)
    described = _member(root, "described")
    for path in (plain, described):
        _write_csv(path.parent / "data")
    project = Project(root)
    failures, _out = _quiet(project.run_all, make_reports=False)
    assert failures == []

    ## With no about: and with a full one: the same tables and the same record
    ## of the settings they were computed with.
    tables = _tables(plain.parent)
    assert "feeding_summary.csv" in tables
    assert _tables(described.parent) == tables
    record = "analysis_settings.json"
    assert (described.parent / "analysis" / record).read_bytes() == \
        (plain.parent / "analysis" / record).read_bytes()

    ## Edit the title, a level's meaning, a level's term and run: nothing is
    ## reported as needing a re-run, and the cached summary is still the one.
    cached = sorted(p.name for p in (described.parent / ".pyflic_cache").iterdir())
    assert cached
    config = _read(described)
    config["about"] = _edited(config["about"])
    _write(described, config)
    stamp = os.path.getmtime(described.parent / "analysis" / "feeding_summary.csv") + 5
    os.utime(described, (stamp, stamp))                      # saved after the analysis
    status = Project(root).member_status("described")
    assert status["analyzed"] is True
    assert status["stale"] is False and status["stale_reasons"] == []
    lines: list[str] = []
    _quiet(Project(root).run_all, make_reports=False, skip_analyzed=True,
           log=lines.append)
    assert any("described" in line and "already analyzed" in line for line in lines)
    assert not any("re-run needed" in line for line in lines)

    ## Run again all the same.  Both members are now served the summary they
    ## cached, so the edited one is compared with the one that has no about:,
    ## run the same way: a cached summary is read back from CSV, and differs
    ## from a computed one in a float's last digit whatever the config says.
    failures, out = _quiet(Project(root).run_all, make_reports=False)
    assert failures == []
    assert out.count("Loaded feeding summary from disk cache.") == 2
    assert _tables(described.parent) == _tables(plain.parent)
    assert sorted(p.name for p in (described.parent / ".pyflic_cache").iterdir()) == cached
    pd.testing.assert_frame_equal(
        pd.read_csv(io.BytesIO(_tables(described.parent)["feeding_summary.csv"])),
        pd.read_csv(io.BytesIO(tables["feeding_summary.csv"])))

    ## And with the section taken away again.
    config.pop("about")
    _write(described, config)
    assert Project(root).member_status("described")["stale"] is False


def test_a_standalone_recordings_summary_ignores_about(tmp_path):
    with_about = _standalone(tmp_path / "a")
    without = _standalone(tmp_path / "b", about=None)
    summaries = []
    for path in (with_about, without):
        _write_csv(path.parent / "data")
        exp, _out = _quiet(load_experiment_yaml, path.parent)
        summaries.append(exp.feeding_summary())
    pd.testing.assert_frame_equal(summaries[0], summaries[1])


def test_editing_run_or_a_levels_term_keeps_the_cache_key(tmp_path):
    path = _standalone(tmp_path / "exp")
    key = cache._config_hash(path.parent)
    config = _read(path)

    config["about"]["run"] = "R-20261122-scott-07"
    _write(path, config)
    assert cache._config_hash(path.parent) == key
    config["about"]["factors"]["Genotype"]["levels"]["Ctrl"]["term"] = "GEN-0099"
    _write(path, config)
    assert cache._config_hash(path.parent) == key
    del config["about"]["run"]                               # not filed after all
    del config["about"]["factors"]["Genotype"]["levels"]["Ctrl"]["term"]
    _write(path, config)
    assert cache._config_hash(path.parent) == key
    ## The whole key, which the cached summary's file name is.
    assert cache.feeding_summary_key(path.parent, range_minutes=(0, 0),
                                     transform_licks=False).split("_")[1] == key

    config["dfms"][0]["chambers"][1] = "Exp"                 # a setting does change it
    _write(path, config)
    assert cache._config_hash(path.parent) != key


# ---------------------------------------------------------------------------
# Never an error: the linter
# ---------------------------------------------------------------------------

def _about_issues(path: Path) -> list:
    return [issue for issue in lint_flic_config(path)
            if (issue.path or "").startswith("about")]


def test_the_linter_is_silent_on_what_a_catalogue_writes(tmp_path):
    """run:, a level mapping's term:, label: and role:, and keys pyflic does
    not know, in the section, in a factor or in a level's mapping."""
    about = copy.deepcopy(ABOUT)
    about["factors"]["Genotype"]["x_factor"] = "kept"
    path = _standalone(tmp_path / "exp", about)
    assert _about_issues(path) == []
    assert not any("unknown top-level key" in issue.message
                   for issue in lint_flic_config(path))


ODD = {
    "a text field that is a list": ({"purpose": ["one", "two"]}, "about.purpose"),
    "a run that is a mapping": ({"run": {"id": 7}}, "about.run"),
    "a sex that is a number": ({"sex": 3}, "about.sex"),
    "a protocol that is a list": ({"protocol": ["FLIC PR"]}, "about.protocol"),
    "factors that are text": ({"factors": "Genotype"}, "about.factors"),
    "a factor that is text": ({"factors": {"Genotype": "strain"}}, "about.factors.Genotype"),
    "levels that are a list": ({"factors": {"Genotype": {"levels": ["Ctrl"]}}},
                               "about.factors.Genotype.levels"),
    "an unknown kind": ({"factors": {"Genotype": {"kind": "strain"}}},
                        "about.factors.Genotype.kind"),
    "a title over 60 characters": ({"title": "x" * 61}, "about.title"),
    "a title of two lines": ({"title": "one\ntwo"}, "about.title"),
    "a factor the design does not declare": ({"factors": {"Dose": {"kind": "treatment"}}},
                                             "about.factors.Dose"),
    "a level the design does not declare": (
        {"factors": {"Genotype": {"levels": {"Other": "not here"}}}},
        "about.factors.Genotype.levels.Other"),
    "a section that is not a mapping": ("see the lab book", "about"),
}


@pytest.mark.parametrize("case", sorted(ODD))
def test_each_odd_thing_in_about_is_a_warning_and_the_config_still_loads(tmp_path, case):
    about, where = ODD[case]
    path = _standalone(tmp_path / "exp", about)
    _write_csv(path.parent / "data", minutes=1)
    issues = _about_issues(path)
    assert [issue.path for issue in issues] == [where], case
    assert [issue.severity for issue in issues] == ["warning"]
    exp, _out = _quiet(load_experiment_yaml, path.parent, eager=False, use_disk_cache=False)
    assert sorted(exp.dfms) == [1]


def _everything_odd() -> dict:
    """One section holding every odd thing that fits in one."""
    return {"run": {"id": 7}, "title": "x" * 61, "purpose": ["one", "two"], "sex": 3,
            "protocol": ["FLIC PR"],
            "factors": {"Genotype": {"kind": "strain", "levels": {"Other": "not here"}},
                        "Dose": {"kind": "treatment"}}}


def test_an_odd_about_does_not_stop_an_analysis(tmp_path):
    root = tmp_path / "Proj"
    plain = _member(root, "plain", about=None)
    odd = _member(root, "odd", about=_everything_odd())
    for path in (plain, odd):
        _write_csv(path.parent / "data")
    assert len(_about_issues(odd)) >= 7
    project = Project(root)                                  # never a load error
    failures, _out = _quiet(project.run_all, make_reports=False)
    assert failures == []
    assert _tables(odd.parent) == _tables(plain.parent)


# ---------------------------------------------------------------------------
# Kept and fresh: the writers that are not windows
# ---------------------------------------------------------------------------

def _loaded(path: Path):
    """A stand-in for the experiment the Hub holds, read when the member was
    loaded: the Selection reads only where it is and its global:."""
    config = _read(path)
    return SimpleNamespace(experiment_dir=str(path.parent), config=config,
                           global_config=dict(config.get("global") or {}))


def test_a_selection_tick_keeps_about_and_writes_the_one_now_on_disk(tmp_path):
    path = _standalone(tmp_path / "exp")
    experiment = _loaded(path)
    assert sel.set_included(experiment, sel.PLOTS, "dot_plot", False) == path
    saved = _read(path)
    assert saved["global"]["omit"] == {"plots": ["dot_plot"]}
    assert saved["about"] == ABOUT

    ## A catalogue rewrites the section while the Hub still holds the member.
    saved["about"] = _edited(ABOUT)
    _write(path, saved)
    sel.set_included(experiment, sel.PLOTS, "dot_plot", True)
    saved = _read(path)
    assert "omit" not in saved["global"]
    assert saved["about"] == _edited(ABOUT)


def test_a_members_own_omit_is_removed_without_touching_its_about(tmp_path):
    """A tick on a Member is written to the Project's design, and a Member
    stating an omit: of its own has it removed so that it inherits."""
    root = tmp_path / "Proj"
    path = _member(root, "rep1", global_={"omit": {"plots": ["dot_plot"]}})
    config = _read(path)
    config["global"] = config.pop("global_")
    _write(path, config)
    sel.set_included(_loaded(path), sel.QC, "bleeding", False)
    assert _read(root / "project.yaml")["design"]["global"]["omit"] == {"qc": ["bleeding"]}
    saved = _read(path)
    assert "global" not in saved
    assert saved["about"] == ABOUT


def test_saving_a_design_into_its_members_keeps_their_about(tmp_path):
    root = tmp_path / "Proj"
    path = _member(root, "rep1")
    assert project_mod.adopt_design(root) == ["rep1"]
    saved = _read(path)
    assert saved["global"] == project_mod.design_copy(DESIGN)
    assert saved["about"] == ABOUT
    assert list(saved["about"]) == list(ABOUT)


def test_a_scaffolded_member_takes_the_shared_about_and_leaves_the_first_alone(tmp_path):
    root = tmp_path / "Proj"
    first = _member(root, "rep1")
    before = first.read_bytes()
    (root / "rep2" / "data").mkdir(parents=True)
    (root / "rep2" / "data" / "DFM1_0.csv").write_text("Sample,Seconds\n")
    path, _notes = Project(root).scaffold_member("rep2")
    about = _read(Path(path))["about"]
    assert about == {key: ABOUT[key] for key in about_mod.SHARED_KEYS}
    assert not {"run", "title", "purpose", "notes"} & set(about)
    assert first.read_bytes() == before


# ---------------------------------------------------------------------------
# Kept and fresh: the windows
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def no_dialogs(monkeypatch):
    """Every box answers at once, so nothing blocks headless."""
    from PyQt6.QtWidgets import QMessageBox

    for name in ("question", "warning", "information", "critical"):
        monkeypatch.setattr(QMessageBox, name,
                            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes))


def _save_in_the_experiment_editor(path: Path) -> None:
    from pyflic.base.experiment_editor import ExperimentEditor

    editor = ExperimentEditor(path)
    try:
        assert editor.write(path, confirm=False)
    finally:
        editor.close()


def _save_in_the_config_editor(path: Path) -> None:
    from pyflic.base.config_editor import FLICConfigEditor

    editor = FLICConfigEditor(initial_path=path)
    try:
        editor._confirm_save_with_problems = lambda _problems: True
        editor._save()
    finally:
        editor.close()


def _save_in_the_script_editor(path: Path) -> None:
    from pyflic.base.script_editor.window import ScriptEditorWindow

    window = ScriptEditorWindow(path)
    try:
        window._scripts.append({"name": "added", "steps": [{"action": "load"}]})
        window._save_to_disk()
    finally:
        window._dirty = False
        window.close()


WINDOWS = {"the Experiment Editor": _save_in_the_experiment_editor,
           "the Config Editor": _save_in_the_config_editor,
           "the Script Editor": _save_in_the_script_editor}


@pytest.mark.parametrize("window", sorted(WINDOWS))
@pytest.mark.parametrize("kind", ["standalone", "member"])
def test_about_is_equal_after_each_window_has_saved_the_config(
        app, no_dialogs, tmp_path, window, kind):
    path = _standalone(tmp_path / "exp") if kind == "standalone" \
        else _member(tmp_path / "Proj", "rep1")
    WINDOWS[window](path)
    saved = _read(path)
    assert saved["about"] == ABOUT, window
    assert list(saved["about"])[0] == "run"
    ## ... and the save was a real one.
    if window == "the Script Editor":
        assert [s["name"] for s in saved["scripts"]] == ["added"]
    elif kind == "member":
        assert saved["global"] == project_mod.design_copy(DESIGN)


def test_about_is_equal_after_every_writer_in_turn(app, no_dialogs, tmp_path):
    """One file, through every writer of an existing config."""
    root = tmp_path / "Proj"
    path = _member(root, "rep1")
    project_mod.adopt_design(root)
    for save in WINDOWS.values():
        save(path)
    sel.set_included(_loaded(path), sel.ANALYSES, "event_statistics", False)
    assert _read(path)["about"] == ABOUT
    assert list(_read(path)) == ["about", "global", "dfms", "scripts"]


def test_the_script_editor_writes_the_about_now_on_disk(app, no_dialogs, tmp_path):
    from pyflic.base.script_editor.window import ScriptEditorWindow

    path = _standalone(tmp_path / "exp")
    window = ScriptEditorWindow(path)
    try:
        config = _read(path)
        config["about"] = _edited(ABOUT)                     # filed while it was open
        _write(path, config)
        window._scripts.append({"name": "added", "steps": [{"action": "load"}]})
        window._save_to_disk()
    finally:
        window._dirty = False
        window.close()
    assert _read(path)["about"] == _edited(ABOUT)


@pytest.mark.parametrize("kind", ["standalone", "member"])
def test_the_config_editor_writes_the_about_now_on_disk(app, no_dialogs, tmp_path, kind):
    from pyflic.base.config_editor import FLICConfigEditor

    path = _standalone(tmp_path / "exp") if kind == "standalone" \
        else _member(tmp_path / "Proj", "rep1")
    editor = FLICConfigEditor(initial_path=path)
    try:
        config = _read(path)
        config["about"] = _edited(ABOUT)                     # filed while it was open
        _write(path, config)
        editor._confirm_save_with_problems = lambda _problems: True
        editor._save()
    finally:
        editor.close()
    assert _read(path)["about"] == _edited(ABOUT)


@pytest.mark.parametrize("window", ["the Experiment Editor", "the Config Editor"])
def test_an_odd_about_does_not_stop_an_editor_saving(app, no_dialogs, tmp_path, window):
    path = _standalone(tmp_path / "exp", _everything_odd())
    _write_csv(path.parent / "data", minutes=1)
    WINDOWS[window](path)
    assert "about" in _read(path)
    exp, _out = _quiet(load_experiment_yaml, path.parent, eager=False, use_disk_cache=False)
    assert sorted(exp.dfms) == [1]

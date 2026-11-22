"""Where pyflic keeps a user's settings (``ui.json``, the type library, ``.env``).

One rule, asked on every call: ``$XDG_CONFIG_HOME/pyflic``, or
``~/.config/pyflic`` when the variable is unset *or empty* (the XDG spec
treats an empty value as unset).  The path used to be fixed when
``ui.settings`` was imported, so a test that set the variable afterwards
still wrote the user's own file; and an empty variable made it the relative
``pyflic/`` of whatever folder the app was started in.
"""

from __future__ import annotations

from pathlib import Path

from pyflic.base import utils
from pyflic.base.experiment_types import library
from pyflic.base.ui import settings


def test_the_folder_follows_the_variable_on_every_call(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "one"))
    settings.set_value("theme", "dark")
    assert (tmp_path / "one" / "pyflic" / "ui.json").is_file()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "two"))
    assert settings.get("theme") == "auto"
    settings.add_recent_project(tmp_path)
    assert (tmp_path / "two" / "pyflic" / "ui.json").is_file()
    assert library.default_library_dir() == tmp_path / "two" / "pyflic" / "experiment_types"


def test_an_empty_variable_is_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", "")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    expected = tmp_path / "home" / ".config" / "pyflic"
    assert utils.user_config_dir() == expected
    assert settings.config_file() == expected / "ui.json"
    assert library.default_library_dir() == expected / "experiment_types"
    settings.set_value("theme", "dark")
    assert (expected / "ui.json").is_file()
    assert not (tmp_path / "pyflic").exists()


def test_a_relative_variable_is_ignored(tmp_path, monkeypatch):
    """The spec requires an absolute path; a relative one would write into
    the current folder, which may be an experiment."""
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert utils.user_config_dir() == Path(tmp_path / "home" / ".config" / "pyflic")

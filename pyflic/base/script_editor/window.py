"""Script Editor main window.

Launched by the config editor via File → Script Editor.  The window is a
non-modal ``QMainWindow`` parented to the config editor so it stacks
naturally without blocking.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..ui import ActionButton, Category, TopBar, apply_theme, icon, resolved_mode
from ..ui import settings as ui_settings
from .actions import validation_issues
from .canvas import Canvas
from .inspector import Inspector
from .palette import Palette
from .preview import Preview


class ScriptEditorWindow(QMainWindow):
    """Visual script editor.

    State model:

    * :pyattr:`_scripts`  — list of script dicts, kept in sync with the
      canvas for the active script.
    * :pyattr:`_active_idx` — index into ``_scripts``; ``-1`` means no
      script selected (empty dropdown).
    * :pyattr:`_dirty`    — True when the in-memory state differs from the
      last thing loaded/saved to disk.
    """

    scriptsSaved = pyqtSignal(Path)

    def __init__(self, config_path: str | Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._config_path = Path(config_path).expanduser().resolve()
        self.setWindowTitle(f"pyflic — Script Editor — {self._config_path.name}")
        self.resize(1400, 860)
        self._scripts: list[dict[str, Any]] = []
        self._active_idx: int = -1
        self._dirty: bool = False
        self._experiment_type: str | None = None

        # ── Build UI ────────────────────────────────────────────────────
        self._build_ui()

        # ── Initial load ────────────────────────────────────────────────
        self._load_from_disk()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Top bar
        self._top_bar = TopBar("Script Editor")
        self._top_bar.add_right(self._build_script_switcher())
        self._btn_theme = QToolButton()
        self._btn_theme.setIcon(icon("theme_dark" if resolved_mode() == "light" else "theme_light"))
        self._btn_theme.setIconSize(QSize(18, 18))
        self._btn_theme.setAutoRaise(True)
        self._btn_theme.setToolTip("Toggle light / dark theme")
        self._btn_theme.clicked.connect(self._toggle_theme)
        self._top_bar.add_right(self._btn_theme)
        outer.addWidget(self._top_bar)

        # Subtitle bar (file path + dirty indicator)
        sub = QFrame()
        sub.setStyleSheet(
            "background: palette(alternate-base); "
            "border-bottom: 1px solid palette(mid);"
        )
        sub_lay = QHBoxLayout(sub)
        sub_lay.setContentsMargins(16, 4, 16, 4)
        self._path_lbl = QLabel(f"File: <code>{self._config_path}</code>")
        self._path_lbl.setTextFormat(Qt.TextFormat.RichText)
        sub_lay.addWidget(self._path_lbl)
        sub_lay.addStretch(1)
        self._dirty_lbl = QLabel("")
        self._dirty_lbl.setStyleSheet("color: #f59e0b; font-weight: 600;")
        sub_lay.addWidget(self._dirty_lbl)
        outer.addWidget(sub)

        # Main three-pane splitter
        self._palette = Palette()
        self._canvas = Canvas()
        self._inspector = Inspector()

        self._palette.actionRequested.connect(self._on_palette_action)
        self._canvas.stepSelected.connect(self._on_step_selected)
        self._canvas.stepsChanged.connect(self._on_canvas_changed)
        self._inspector.stepEdited.connect(self._on_inspector_edited)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(4)
        split.addWidget(self._palette)
        split.addWidget(self._canvas)
        split.addWidget(self._inspector)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setStretchFactor(2, 3)
        outer.addWidget(split, 1)

        # Preview + save bar
        bottom = QFrame()
        bot_lay = QVBoxLayout(bottom)
        bot_lay.setContentsMargins(12, 8, 12, 10)
        bot_lay.setSpacing(6)

        self._preview = Preview()
        self._preview.showFullToggled.connect(lambda _v: self._refresh_preview())
        bot_lay.addWidget(self._preview)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        warn = QLabel("Comments and formatting in your YAML file are not preserved on save.")
        warn.setStyleSheet("color: palette(mid); font-size: 9pt;")
        action_row.addWidget(warn)
        action_row.addStretch(1)

        self._btn_reload = ActionButton("Reload from disk", category=Category.TOOLS,
                                        icon_name="open")
        self._btn_reload.clicked.connect(self._reload_from_disk)
        action_row.addWidget(self._btn_reload)

        self._btn_save = ActionButton("Save to config", category=Category.LOAD,
                                      icon_name="save", primary=True)
        self._btn_save.clicked.connect(self._save_to_disk)
        action_row.addWidget(self._btn_save)
        bot_lay.addLayout(action_row)

        outer.addWidget(bottom)

    def _build_script_switcher(self) -> QWidget:
        host = QWidget()
        lay = QHBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        lay.addWidget(QLabel("Script:"))
        self._cmb_script = QComboBox()
        self._cmb_script.setMinimumWidth(200)
        self._cmb_script.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._cmb_script.currentIndexChanged.connect(self._on_script_switched)
        lay.addWidget(self._cmb_script)

        self._btn_new = QToolButton()
        self._btn_new.setIcon(icon("new", category=Category.LOAD))
        self._btn_new.setToolTip("New script")
        self._btn_new.setAutoRaise(True)
        self._btn_new.clicked.connect(self._on_new_script)
        lay.addWidget(self._btn_new)

        self._btn_rename = QToolButton()
        self._btn_rename.setIcon(icon("save_as", category=Category.TOOLS))
        self._btn_rename.setToolTip("Rename current script")
        self._btn_rename.setAutoRaise(True)
        self._btn_rename.clicked.connect(self._on_rename_script)
        lay.addWidget(self._btn_rename)

        self._btn_delete = QToolButton()
        self._btn_delete.setIcon(icon("clear", category=Category.QC))
        self._btn_delete.setToolTip("Delete current script")
        self._btn_delete.setAutoRaise(True)
        self._btn_delete.clicked.connect(self._on_delete_script)
        lay.addWidget(self._btn_delete)

        return host

    # ==================================================================
    # Disk I/O
    # ==================================================================

    def _load_from_disk(self) -> None:
        """Read ``self._config_path`` and populate ``self._scripts``."""
        try:
            raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Read error",
                f"Failed to read YAML file:\n{self._config_path}\n\n{exc}",
            )
            raw = {}

        if not isinstance(raw, dict):
            raw = {}

        scripts_raw = raw.get("scripts") or []
        self._scripts = [
            dict(s) for s in scripts_raw
            if isinstance(s, dict) and s.get("name")
        ]

        # Pull experiment_type so the palette / inspector can surface warnings.
        g = raw.get("global") or {}
        et_raw = (g.get("experiment_type") or "") if isinstance(g, dict) else ""
        self._experiment_type = (
            str(et_raw).strip().lower().replace("-", "_").replace(" ", "_")
            or None
        )
        self._palette.set_experiment_type(self._experiment_type)
        self._canvas.set_experiment_type(self._experiment_type)
        self._inspector.set_experiment_type(self._experiment_type)

        self._rebuild_switcher(preferred_idx=0 if self._scripts else -1)
        self._set_dirty(False)

    def _reload_from_disk(self) -> None:
        if self._dirty:
            ans = QMessageBox.question(
                self, "Discard changes?",
                "You have unsaved changes that will be discarded.\n\nReload anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        self._load_from_disk()

    def _save_to_disk(self) -> None:
        # Capture the current script back into self._scripts.
        self._commit_canvas_to_model()

        # Read the full file, replace `scripts`, write back.
        try:
            raw = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Read error",
                f"Failed to re-read YAML before save:\n{self._config_path}\n\n{exc}",
            )
            return
        if not isinstance(raw, dict):
            raw = {}

        clean_scripts = [self._clean_script(s) for s in self._scripts]
        if clean_scripts:
            raw["scripts"] = clean_scripts
        elif "scripts" in raw:
            del raw["scripts"]

        try:
            self._config_path.write_text(
                yaml.dump(
                    raw, default_flow_style=False, allow_unicode=True, sort_keys=False
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self, "Write error",
                f"Failed to write YAML:\n{self._config_path}\n\n{exc}",
            )
            return

        self._set_dirty(False)
        self.scriptsSaved.emit(self._config_path)
        self.statusBar().showMessage(
            f"Saved {len(clean_scripts)} script(s) to {self._config_path.name}", 4000
        )

    @staticmethod
    def _clean_script(script: dict[str, Any]) -> dict[str, Any]:
        """Return a cleaned copy of *script* suitable for YAML dump.

        * Drops empty / None fields (so the yaml stays tidy).
        * Ensures ``steps`` is a list of dicts (drops empty steps).
        """
        out: dict[str, Any] = {}
        name = script.get("name", "")
        if isinstance(name, str) and name.strip():
            out["name"] = name.strip()
        else:
            out["name"] = "(unnamed)"
        if script.get("start") is not None:
            out["start"] = script["start"]
        if script.get("end") is not None:
            out["end"] = script["end"]
        steps_raw = script.get("steps") or []
        out["steps"] = [
            ScriptEditorWindow._clean_step(s) for s in steps_raw
            if isinstance(s, dict) and s.get("action")
        ]
        return out

    @staticmethod
    def _clean_step(step: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"action": step["action"]}
        for k, v in step.items():
            if k == "action":
                continue
            if v is None or v == "":
                continue
            if isinstance(v, list) and not v:
                continue
            out[k] = v
        return out

    # ==================================================================
    # Script switcher & CRUD
    # ==================================================================

    def _rebuild_switcher(self, *, preferred_idx: int) -> None:
        self._cmb_script.blockSignals(True)
        self._cmb_script.clear()
        for s in self._scripts:
            self._cmb_script.addItem(s.get("name", "(unnamed)"))
        if preferred_idx < 0 or preferred_idx >= len(self._scripts):
            self._active_idx = -1
            self._canvas.load_script(None)
            self._inspector.clear()
        else:
            self._cmb_script.setCurrentIndex(preferred_idx)
            self._active_idx = preferred_idx
            self._canvas.load_script(self._scripts[preferred_idx])
        self._cmb_script.blockSignals(False)
        self._refresh_preview()

    def _on_script_switched(self, idx: int) -> None:
        # Persist the current canvas state into the model before switching.
        self._commit_canvas_to_model()
        self._active_idx = idx
        if idx < 0:
            self._canvas.load_script(None)
            self._inspector.clear()
        else:
            self._canvas.load_script(self._scripts[idx])
        self._refresh_preview()

    def _on_new_script(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New script", "Script name:", text="untitled"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(s.get("name") == name for s in self._scripts):
            QMessageBox.warning(self, "Duplicate name",
                                f"A script named {name!r} already exists.")
            return
        self._commit_canvas_to_model()
        self._scripts.append({"name": name, "steps": []})
        self._rebuild_switcher(preferred_idx=len(self._scripts) - 1)
        self._set_dirty(True)

    def _on_rename_script(self) -> None:
        if self._active_idx < 0:
            return
        current = self._scripts[self._active_idx].get("name", "")
        name, ok = QInputDialog.getText(
            self, "Rename script", "New name:", text=str(current)
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name == current:
            return
        if any(i != self._active_idx and s.get("name") == name
               for i, s in enumerate(self._scripts)):
            QMessageBox.warning(self, "Duplicate name",
                                f"A script named {name!r} already exists.")
            return
        self._commit_canvas_to_model()
        self._scripts[self._active_idx]["name"] = name
        self._rebuild_switcher(preferred_idx=self._active_idx)
        self._set_dirty(True)

    def _on_delete_script(self) -> None:
        if self._active_idx < 0:
            return
        name = self._scripts[self._active_idx].get("name", "")
        ans = QMessageBox.question(
            self, "Delete script?",
            f"Delete script {name!r}? This cannot be undone until you "
            f"click Reload from disk.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans != QMessageBox.StandardButton.Yes:
            return
        del self._scripts[self._active_idx]
        new_idx = min(self._active_idx, len(self._scripts) - 1)
        self._rebuild_switcher(preferred_idx=new_idx)
        self._set_dirty(True)

    # ==================================================================
    # Canvas / palette / inspector wiring
    # ==================================================================

    def _on_palette_action(self, action_name: str) -> None:
        if self._active_idx < 0:
            # User hit an action tile before creating a script → create one.
            self._on_new_script()
            if self._active_idx < 0:
                return
        self._canvas.append_step(action_name)

    def _on_step_selected(self, row: int) -> None:
        if row < 0:
            self._inspector.clear()
        else:
            self._inspector.show_step(self._canvas.selected_step())

    def _on_canvas_changed(self) -> None:
        # Canvas is authoritative for the active script; sync into model.
        self._commit_canvas_to_model()
        # Re-sync the combobox label if the name changed.
        if self._active_idx >= 0:
            new_name = self._scripts[self._active_idx].get("name", "")
            current_item_text = self._cmb_script.itemText(self._active_idx)
            if new_name != current_item_text:
                self._cmb_script.blockSignals(True)
                self._cmb_script.setItemText(self._active_idx, new_name)
                self._cmb_script.blockSignals(False)
        self._set_dirty(True)
        self._refresh_preview()

    def _on_inspector_edited(self, step: dict[str, Any]) -> None:
        self._canvas.update_selected_step(step)

    def _commit_canvas_to_model(self) -> None:
        if self._active_idx < 0:
            return
        self._scripts[self._active_idx] = self._canvas.current_script()

    # ==================================================================
    # Dirty-state + preview
    # ==================================================================

    def _set_dirty(self, dirty: bool) -> None:
        self._dirty = dirty
        self._dirty_lbl.setText("● unsaved changes" if dirty else "")
        self._btn_save.setEnabled(dirty)

    def _refresh_preview(self) -> None:
        if self._preview.is_show_full():
            self._preview.set_scripts_block(
                [self._clean_script(s) for s in self._scripts]
            )
        else:
            if self._active_idx < 0:
                self._preview.set_script({"name": "(no script)", "steps": []})
            else:
                self._preview.set_script(
                    self._clean_script(self._canvas.current_script())
                )

    # ==================================================================
    # Theme toggle
    # ==================================================================

    def _toggle_theme(self) -> None:
        from ..ui import theme as _theme

        new_mode = "light" if _theme.resolved_mode() == "dark" else "dark"
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode=new_mode)
        ui_settings.set_value("theme", new_mode)
        self._btn_theme.setIcon(
            icon("theme_dark" if _theme.resolved_mode() == "light" else "theme_light")
        )

    # ==================================================================
    # Close handling
    # ==================================================================

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self._dirty:
            ans = QMessageBox.question(
                self, "Unsaved changes",
                "You have unsaved script changes. Save before closing?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
            )
            if ans == QMessageBox.StandardButton.Save:
                self._save_to_disk()
                if self._dirty:
                    event.ignore()
                    return
            elif ans == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        super().closeEvent(event)


def launch(config_path: str | Path) -> None:
    """Stand-alone launcher for development.

    Usually the config editor instantiates :class:`ScriptEditorWindow`
    directly; this entry point makes it easy to smoke-test in isolation.
    """
    import sys

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    win = ScriptEditorWindow(Path(config_path))
    win.show()
    raise SystemExit(app.exec())

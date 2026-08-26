"""The Batch Run preflight (ADR-0009).

Recursive discovery means the folder you picked no longer tells you what will
run: Projects can sit at any depth, and a Batch Run rewrites analysis in every
one of them.  This dialog is the one surface that states the target list, and
the last point at which it can be changed.

It shows, per Project, its relative-path key, how many Members the run can
actually use, and every **Blocked Member** with the reason and the action that
clears it — filing an Unfiled Recording, or scaffolding a config through the
Project's own Member configs… dialog.  It also previews the Exclusion Sheet:
what each row would do, and one switch to decline it for this run without
touching the sheet or any standing declaration (ADR-0010).

Nothing here is a gate.  A Project with blocked Members still runs its healthy
ones, and the run is never refused — a stale folder must not stop ten Projects
at 2am.
"""

from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from . import batch as batch_mod
from . import layout as layout_mod
from .ui import Category, icon

#: Roles on a tree row: which Project, and which Member directory (if any).
_KEY_ROLE = Qt.ItemDataRole.UserRole
_DIR_ROLE = Qt.ItemDataRole.UserRole + 1

# Project keys are relative paths in recursive batches.  Keep that first column
# bounded so a deep Project cannot force the Batch surfaces wider than their
# panels; Qt paints the hidden tail with an ellipsis.
PROJECT_COLUMN_MAX_WIDTH = 280


def blocked_brush() -> QBrush:
    """The Blocked-Member red as a brush — one definition, in the theme."""
    from .ui import blocked_color

    return QBrush(QColor(blocked_color()))


class BatchPreflightDialog(QDialog):
    """Confirm (and repair) what a Batch Run is about to do.

    ``exec()`` returning ``Accepted`` means run; :attr:`selected_keys` and
    :attr:`apply_exclusions` then carry the confirmed target list and whether
    the Exclusion Sheet was declined.
    """

    def __init__(self, parent, root, *, checked=None, log=None) -> None:
        super().__init__(parent)
        self._root = str(root)
        self._log = log or (lambda _text: None)
        self._preferred = None if checked is None else {str(k) for k in checked}
        self._projects: list = []
        self._skipped: list = []
        self._loading = False
        #: Keys the USER unchecked.  A Project with nothing usable starts
        #: unchecked on its own, and on a batch of freshly-arrived recordings
        #: that is every Project — so "do not touch what I unchecked" has to
        #: mean the user's own act, or filing would refuse the one job it
        #: exists for.
        self._user_unchecked: set = set()
        #: Keys the user checked by hand — a Project nothing can run in still
        #: joins the run if they insist.
        self._user_checked: set = set()
        self._seeded = False
        #: Whether the user has taken a view on the exclusion-sheet switch.
        self._sheet_choice = None
        self.setWindowTitle("Batch Run — review")
        self.setMinimumSize(760, 520)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        self._heading = QLabel("")
        self._heading.setWordWrap(True)
        self._heading.setStyleSheet("font-weight: 600;")
        outer.addWidget(self._heading)

        self._tree = QTreeWidget()
        self._tree.setColumnCount(3)
        self._tree.setHeaderLabels(["Project", "Members", "Status"])
        self._tree.setTextElideMode(Qt.TextElideMode.ElideRight)
        self._tree.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection)
        header = self._tree.header()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Interactive)
        self._tree.setColumnWidth(0, PROJECT_COLUMN_MAX_WIDTH)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self._tree.itemDoubleClicked.connect(self._on_double_click)
        outer.addWidget(self._tree, 1)

        actions = QHBoxLayout()
        self._btn_fix = QPushButton("File data…")
        self._btn_fix.setIcon(icon("file", category=Category.TOOLS))
        self._btn_fix.setToolTip(
            "Move the selected member's DFM CSVs into data/ (and any other "
            "loose file into extra_files/).  Never overwrites.")
        self._btn_fix.clicked.connect(self._fix_selected)
        self._btn_fix.setEnabled(False)
        actions.addWidget(self._btn_fix)
        self._btn_fix_all = QPushButton("File every unfiled recording")
        self._btn_fix_all.setToolTip(
            "File every member whose DFM CSVs sit loose at its root.  "
            "Ambiguous ones (the same DFM both loose and filed) are left "
            "alone and reported.")
        self._btn_fix_all.clicked.connect(self._fix_all)
        actions.addWidget(self._btn_fix_all)
        btn_rescan = QPushButton("Rescan")
        btn_rescan.setToolTip("Walk the batch folder again — for changes made "
                              "outside the app.")
        btn_rescan.clicked.connect(self.reload)
        actions.addWidget(btn_rescan)
        actions.addStretch(1)
        outer.addLayout(actions)

        self._sheet_box = QCheckBox("Apply the exclusion sheet before running")
        self._sheet_box.setChecked(True)
        self._sheet_box.toggled.connect(self._on_sheet_toggled)
        self._sheet_box.setToolTip(
            "Write the sheet's rows into each member's remove_chambers.csv.  "
            "Unchecking skips it for this run only — the sheet and every "
            "standing declaration are left untouched.")
        outer.addWidget(self._sheet_box)
        self._sheet_label = QLabel("")
        self._sheet_label.setWordWrap(True)
        self._sheet_label.setStyleSheet("color: palette(mid);")
        outer.addWidget(self._sheet_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._btn_run = buttons.addButton(
            "Run batch", QDialogButtonBox.ButtonRole.AcceptRole)
        self._btn_run.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._tree.currentItemChanged.connect(
            lambda *_a: self._sync_action_buttons())
        ## Checking a Project changes both which sheet rows are in scope and
        ## what "file everything" would touch.
        self._tree.itemChanged.connect(self._on_item_changed)
        self.reload()

    # ------------------------------------------------------------------
    # State the caller reads back
    # ------------------------------------------------------------------

    @property
    def selected_keys(self) -> list[str]:
        """Project keys the user left checked, in run order."""
        keys = []
        for index in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(index)
            if item.checkState(0) == Qt.CheckState.Checked:
                keys.append(item.data(0, _KEY_ROLE))
        return keys

    @property
    def apply_exclusions(self) -> bool:
        return self._sheet_box.isChecked()

    @property
    def projects(self) -> list:
        return list(self._projects)

    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Re-walk the batch folder and rebuild the tree, preserving checks."""
        self._loading = True
        try:
            self._reload()
        finally:
            self._loading = False

    def _reload(self) -> None:
        found = batch_mod.discover(self._root)
        self._projects = found["projects"]
        self._skipped = found["skipped"]

        if self._preferred is not None and not self._seeded:
            ## The Hub's own check column, but only its EXPLICIT half: a row
            ## unchecked there because nothing in it could run is not the user
            ## saying "never run this", and treating it as such is what keeps a
            ## Project excluded after the preflight has just repaired it.
            self._seeded = True
            for item in self._projects:
                if item.runnable and item.key not in self._preferred:
                    self._user_unchecked.add(item.key)
                if not item.runnable and item.key in self._preferred:
                    self._user_checked.add(item.key)

        self._tree.clear()
        for project in self._projects:
            item = QTreeWidgetItem(
                [project.key,
                 f"{len(project.usable)}/{len(project.members)}", ""])
            item.setData(0, _KEY_ROLE, project.key)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            ## Derived fresh every time from what the user has actually said
            ## plus what the Project can do NOW — never from the previous check
            ## column.  A Project with nothing usable starts unchecked (it can
            ## only produce a failure) and becomes checked the moment filing
            ## makes it runnable, which is the whole point of repairing it here.
            checked = (project.key in self._user_checked
                       or (project.runnable
                           and project.key not in self._user_unchecked))
            item.setCheckState(0, Qt.CheckState.Checked if checked
                               else Qt.CheckState.Unchecked)
            blocked = project.blocked
            if blocked:
                item.setText(2, f"{len(blocked)} blocked")
                item.setForeground(2, blocked_brush())
            elif not project.members:
                item.setText(2, "no members")
            else:
                item.setText(2, "ok")
            for member in blocked:
                child = QTreeWidgetItem([member.name, "", member.status])
                child.setData(0, _KEY_ROLE, project.key)
                child.setData(0, _DIR_ROLE, member.directory)
                child.setForeground(2, blocked_brush())
                child.setToolTip(0, member.detail)
                child.setToolTip(2, member.detail)
                item.addChild(child)
            self._tree.addTopLevelItem(item)
            ## After it joins the tree: Qt ignores setExpanded on an item that
            ## has no view yet, which leaves every blocked reason collapsed
            ## behind an arrow nobody clicks.
            item.setExpanded(bool(blocked))

        if found.get("truncated"):
            self._log("[batch] WARNING: the scan stopped early — this folder "
                      "is larger than a batch should be, and projects deeper "
                      "in it were not found.")
        self._refresh_heading()
        self._refresh_sheet()
        self._sync_action_buttons()

    def _refresh_heading(self) -> None:
        chosen = set(self.selected_keys)
        running = [p for p in self._projects if p.key in chosen]
        members = sum(len(p.usable) for p in running)
        blocked = sum(len(p.blocked) for p in self._projects)
        text = (f"{len(running)} of {len(self._projects)} project(s), "
                f"{members} member(s) will run — {self._root}")
        if blocked:
            text += f" · {blocked} blocked member(s) listed below"
        self._heading.setText(text)
        for key, why in self._skipped:
            ## Each carries its own reason — a stray marker, a symlink, an
            ## unreadable folder — and a run that quietly found fewer projects
            ## than the user expects is the failure recursion introduces.
            self._log(f"[batch] {key} skipped — {why}")

    def _refresh_sheet(self) -> None:
        preview = batch_mod.preview_exclusion_sheet(
            self._root, projects=self.selected_keys)
        sheet = preview.get("sheet")
        if sheet is None:
            self._sheet_box.setVisible(False)
            self._sheet_label.setText(
                "No exclusion sheet in this batch folder.")
            return
        self._sheet_box.setVisible(True)
        if preview.get("error"):
            self._sheet_label.setText(
                f"{os.path.basename(sheet)} could not be read: "
                f"{preview['error']}")
            self._sheet_box.setChecked(False)
            self._sheet_box.setEnabled(False)
            return
        ## Re-armed once the sheet reads again — latching it off means
        ## repairing the sheet and rescanning silently runs without it.
        self._sheet_box.setEnabled(True)
        if self._sheet_choice is None:
            self._sheet_box.setChecked(True)
        counts = preview.get("counts", {})
        parts = [f"{status}: {n}" for status, n in sorted(counts.items())]
        skipped = preview.get("skipped") or 0
        if skipped:
            parts.append(f"{skipped} row(s) for projects not in this run")
        self._sheet_label.setText(
            f"{os.path.basename(sheet)} — " + (", ".join(parts) or "no rows"))
        self._sheet_label.setToolTip(
            "\n".join(result.describe()
                      for result in preview.get("results", [])[:40]))

    def _sync_action_buttons(self) -> None:
        item = self._tree.currentItem()
        directory = item.data(0, _DIR_ROLE) if item is not None else None
        fixable = False
        if directory:
            fixable = layout_mod.plan_filing(directory).possible
        self._btn_fix.setEnabled(bool(fixable))
        self._btn_fix_all.setEnabled(bool(self._unfiled()))

    def _unfiled(self) -> list:
        """Unfiled Recordings filing may touch.

        Everything listed EXCEPT what the user explicitly unchecked: moving
        files inside a colleague's Project that was deliberately excluded is
        the same violation as writing an Exclusion Sheet into one, while
        refusing to file a Project that is unchecked only *because* nothing in
        it runs yet would break the common case — a batch of recordings
        straight off the rig, where nothing runs until it is filed.
        """
        return [member for project in self._projects
                if project.key not in self._user_unchecked
                for member in project.blocked
                if member.fix == "file"]

    def _on_item_changed(self, item, column: int) -> None:
        if column or self._loading or item.parent() is not None:
            return
        key = item.data(0, _KEY_ROLE)
        if item.checkState(0) == Qt.CheckState.Checked:
            self._user_unchecked.discard(key)
            self._user_checked.add(key)
        else:
            self._user_checked.discard(key)
            self._user_unchecked.add(key)
        ## Checking a Project changes the sheet's scope, what filing would
        ## touch, and the member counts the heading states.
        self._refresh_heading()
        self._refresh_sheet()
        self._sync_action_buttons()

    def _on_sheet_toggled(self, on: bool) -> None:
        if not self._loading:
            self._sheet_choice = on

    def focus_project(self, key) -> None:
        """Select (and expand) Project *key* — used when the preflight is
        opened from a specific row's right-click."""
        for index in range(self._tree.topLevelItemCount()):
            item = self._tree.topLevelItem(index)
            if item.data(0, _KEY_ROLE) == key:
                item.setExpanded(True)
                self._tree.setCurrentItem(
                    item.child(0) if item.childCount() else item)
                self._tree.scrollToItem(item)
                return

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_double_click(self, item, _column: int) -> None:
        directory = item.data(0, _DIR_ROLE)
        if not directory:
            return
        member = layout_mod.classify(directory)
        if member.fix == "file":
            self._file(directory)
        elif member.fix == "config":
            self._scaffold(item.data(0, _KEY_ROLE))
        else:
            QMessageBox.information(
                self, "Blocked member",
                f"{member.name}: {member.detail or member.status}\n\n"
                "Nothing here can be fixed automatically.")

    def _fix_selected(self) -> None:
        item = self._tree.currentItem()
        if item is not None and item.data(0, _DIR_ROLE):
            self._file(item.data(0, _DIR_ROLE))

    def _fix_all(self) -> None:
        targets = self._unfiled()
        if not targets:
            return
        confirm = QMessageBox.question(
            self, "File every unfiled recording",
            f"Move the DFM CSVs into data/ in {len(targets)} member "
            "director(ies) of the checked projects?\n\nEvery other loose file "
            "goes to extra_files/.  YAML files and an exclusion sheet stay "
            "where they are, and nothing is overwritten.")
        if confirm != QMessageBox.StandardButton.Yes:
            return
        for member in targets:
            self._file(member.directory, reload=False)
        self.reload()

    def _file(self, directory, reload: bool = True) -> None:
        plan = layout_mod.file_recording(directory, log=self._log)
        name = os.path.basename(str(directory))
        if plan.refused:
            self._log(f"[file] {name}: {plan.refused}")
            QMessageBox.warning(self, "Cannot file this member",
                                f"{name}: {plan.refused}")
        else:
            self._log(f"[file] {name}: {plan.describe()}")
        for skipped_name, why in plan.skipped:
            self._log(f"[file] {name}: {skipped_name} skipped — {why}")
        if reload:
            self.reload()

    def _scaffold(self, key) -> None:
        """Hand off to the Project's own Member configs… dialog — the one
        design-aware scaffolding path.

        It is parented to the Hub rather than to this dialog: it needs the Hub
        to launch the Config Editor, and loading the Project can raise (a
        design mismatch is exactly the kind of Project someone batches).
        """
        from . import project as project_mod
        from .hub import MemberConfigsDialog

        directory = batch_mod.project_directory(self._root, key)
        try:
            project = project_mod.Project(str(directory))
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(
                self, "Member configs",
                f"{key} could not be opened as a Project:\n{err}")
            return
        MemberConfigsDialog(self.parent() or self, project).exec()
        self.reload()

"""The Project editor: ``project.yaml``'s name, notes and **Design**.

A Project's authority over its Members is only as good as somebody's ability to
state it.  Before this dialog a Project could be created but its ``design:``
section had to be hand-written, so the common path produced a Project with no
Design at all — Members then validated against *each other* (the legacy mode)
and the first one to be edited silently became the standard.

Everything the Design owns (ADR-0005 — experiment type, detection ``params:``,
``well_names``, ``transform_licks``, ``constants:``, the design factors, the
facet cutoffs) is authored here, in one place, and every Member inherits it.
The parameter form and the factors table are the config editor's own widgets,
so the Design is stated in exactly the vocabulary a Member's config uses.

Modelled on PyTrackingAnalysis's ``ProjectInfoDialog``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import experiment_types, project as project_mod
from .config_editor import FactorsWidget, ParamsForm
from .ui import Category, icon
from .ui.widgets import Card

#: The three auto-removal cutoffs, in the order the config editor shows them.
_CONSTANT_ROWS: tuple[tuple[str, str, str], ...] = (
    ("min_untransformed_licks_cutoff", "Min Untransformed Licks",
     "e.g. 20  (leave blank to skip)"),
    ("max_med_duration_cutoff", "Max Median Duration",
     "e.g. 13  (leave blank to skip)"),
    ("max_events_cutoff", "Max Events", "e.g. 150  (leave blank to skip)"),
)


def _number(text: str):
    """A yaml-tidy number from *text* — ints stay ints.  Raises ValueError."""
    value = float(text)
    return int(value) if value == int(value) else value


class ProjectDesignDialog(QDialog):
    """Create or edit a Project: its ``project.yaml`` and the Design in it.

    Choosing a directory that already is a Project edits it rather than
    starting again, so there is one dialog rather than a create/edit pair that
    would drift.
    """

    def __init__(self, parent: QWidget | None = None,
                 start_dir: str | Path | None = None, *,
                 initialize_existing: bool = False) -> None:
        super().__init__(parent)
        ## Two of the three ways into a Project share this dialog: creating the
        ## directory outright, and initializing one that is already on disk.
        ## The design half is identical; only the directory and the name
        ## differ, so the mode is a flag, not a subclass.
        self._initialize = bool(initialize_existing)
        self.saved_dir: str | None = None
        #: Members whose own ``global:`` was removed on save so they inherit.
        self.adopted: list[str] = []
        self.setWindowTitle("Initialize existing directory"
                            if self._initialize else "Project design")
        self.setMinimumSize(720, 620)
        ## Window-modal: the Hub behind it is what this edits, but a config
        ## editor or help window opened alongside must stay usable.
        self.setWindowModality(Qt.WindowModality.WindowModal)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        intro = QLabel(
            "Turn a folder you already have into a Project: it keeps its own "
            "name, its subdirectories become the Members, and the design below "
            "is inferred from the first one that already has a config.  This "
            "writes project.yaml into it."
            if self._initialize else
            "A Project is a directory whose subdirectories are its Members — "
            "different experiments addressing one question.  The design below "
            "is written to project.yaml and is the <b>authority</b> for every "
            "member: a member that contradicts it fails to load, and a member "
            "that states nothing inherits it.")
        intro.setWordWrap(True)
        intro_row = QHBoxLayout()
        intro_row.addWidget(intro, 1)
        try:
            from ..help import HelpButton

            intro_row.addWidget(
                HelpButton("concepts-project",
                           tooltip="Projects, members, and the design"),
                0, Qt.AlignmentFlag.AlignTop)
        except Exception:  # noqa: BLE001 - help is optional, the editor is not
            pass
        outer.addLayout(intro_row)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        body_layout.addWidget(self._build_identity_card(start_dir))
        body_layout.addWidget(self._build_design_card())
        body_layout.addWidget(self._build_params_card())
        body_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.dir_edit.textChanged.connect(self._prefill_from_dir)
        self._prefill_from_dir()
        self._on_type_changed()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_identity_card(self, start_dir) -> Card:
        card = Card("Project", Category.LOAD, icon_name="project")
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        row = QHBoxLayout()
        self.dir_edit = QLineEdit(str(start_dir or ""))
        row.addWidget(self.dir_edit, 1)
        browse = QPushButton("Browse…")
        browse.setIcon(icon("open", category=Category.LOAD))
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Directory:", holder)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("defaults to the directory name")
        if self._initialize:
            ## The folder IS the Project, so it names it — shown rather than
            ## hidden, so what it will be called is not a surprise.
            self.name_edit.setReadOnly(True)
            self.name_edit.setToolTip(
                "The chosen folder's own name — initializing in place does "
                "not rename it.")
        form.addRow("Project name:", self.name_edit)

        self.notes_edit = QPlainTextEdit()
        self.notes_edit.setPlaceholderText(
            "Optional — shown near the top of the Project Report.")
        self.notes_edit.setMaximumHeight(70)
        form.addRow("Notes:", self.notes_edit)
        card.add_body(form)
        return card

    def _build_design_card(self) -> Card:
        card = Card("Design", Category.ANALYZE, icon_name="settings",
                    subtitle="Enforced on every member of this Project.")
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        self.type_combo = QComboBox()
        for item in experiment_types.available_experiment_types():
            self.type_combo.addItem(item.display_name, item.name)
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        form.addRow("Experiment type:", self.type_combo)

        self.layout_combo = QComboBox()
        self.layout_combo.addItem("two_well  (6 chambers)", "two_well")
        self.layout_combo.addItem("single_well  (12 chambers)", "single_well")
        self.layout_combo.currentIndexChanged.connect(self._on_layout_changed)
        form.addRow("Chamber layout:", self.layout_combo)
        self.layout_note = QLabel("")
        self.layout_note.setObjectName("PyflicCardSubtitle")
        self.layout_note.setWordWrap(True)
        form.addRow("", self.layout_note)

        self.well_a_edit = QLineEdit()
        self.well_a_edit.setPlaceholderText("e.g. Sucrose")
        self.well_b_edit = QLineEdit()
        self.well_b_edit.setPlaceholderText("e.g. Yeast")
        self._well_a_row = form.rowCount()
        form.addRow("Well A:", self.well_a_edit)
        self._well_b_row = form.rowCount()
        form.addRow("Well B:", self.well_b_edit)

        self.transform_check = QCheckBox(
            "Apply the 0.25-power transform to Licks")
        self.transform_check.setChecked(True)
        form.addRow("Transform licks:", self.transform_check)

        self.cutoffs_edit = QLineEdit()
        self.cutoffs_edit.setPlaceholderText("e.g. 10, 70  (minutes)")
        form.addRow("Facet cutoffs:", self.cutoffs_edit)
        self.labels_edit = QLineEdit()
        self.labels_edit.setPlaceholderText(
            "e.g. Acclimation, Experiment, Cooldown")
        form.addRow("Phase names:", self.labels_edit)

        self.exclusion_edit = QLineEdit()
        self.exclusion_edit.setPlaceholderText("general")
        self.exclusion_edit.setToolTip(
            "The group column every member's remove_chambers.csv is read "
            "under, so one rule filters the whole Project.")
        form.addRow("Exclusion group:", self.exclusion_edit)

        self._form = form
        card.add_body(form)

        card.add_section_label("Design factors")
        hint = QLabel(
            "The factors a chamber assignment names, in this order.  Members "
            "assign levels per chamber; the factors themselves are fixed here.")
        hint.setObjectName("PyflicCardSubtitle")
        hint.setWordWrap(True)
        card.add_body(hint)
        self.factors_widget = FactorsWidget()
        card.add_body(self.factors_widget)
        return card

    def _build_params_card(self) -> Card:
        card = Card("Detection parameters", Category.ANALYZE,
                    subtitle="One detection rule across every member — that is "
                            "what makes the Combined Analysis comparable.")
        self.params_form = ParamsForm(override_mode=False, chamber_size=2,
                                      num_columns=2)
        card.add_body(self.params_form)

        card.add_section_label("Auto-filter thresholds "
                               "(used by auto_remove_chambers)")
        cform = QFormLayout()
        cform.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.constant_edits: dict[str, QLineEdit] = {}
        for key, label, placeholder in _CONSTANT_ROWS:
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMaximumWidth(240)
            self.constant_edits[key] = edit
            cform.addRow(f"{label}:", edit)
        card.add_body(cform)
        return card

    # ------------------------------------------------------------------
    # Reacting to choices
    # ------------------------------------------------------------------

    def _experiment_type(self):
        return experiment_types.get_experiment_type(
            self.type_combo.currentData())

    def _chamber_layout(self) -> str:
        item = self._experiment_type()
        if item.chamber_layout is not None:
            return item.chamber_layout
        return self.layout_combo.currentData() or "two_well"

    def _on_type_changed(self) -> None:
        """Show what the type owns, and seed its defaults into empty fields."""
        item = self._experiment_type()
        fixed_layout = item.chamber_layout is not None
        if fixed_layout:
            index = self.layout_combo.findData(item.chamber_layout)
            self.layout_combo.blockSignals(True)
            self.layout_combo.setCurrentIndex(max(index, 0))
            self.layout_combo.blockSignals(False)
        self.layout_combo.setEnabled(not fixed_layout)
        self.layout_note.setText(
            f"'{item.display_name}' owns the chamber layout — it is derived, "
            f"never written to the yaml." if fixed_layout else
            "A Custom Experiment states its own layout.")

        self.cutoffs_edit.setEnabled(not item.facets_fixed)
        if getattr(item, "data_derived_facets", False):
            ## The windows come from the data, per Chamber Group (ADR-0013);
            ## there is no cutoff to author and none is written.
            self.cutoffs_edit.setText("")
            self.cutoffs_edit.setPlaceholderText(
                "derived from the data (training end per chamber group)")
            self.cutoffs_edit.setToolTip(
                f"'{item.display_name}' derives its facets from each chamber "
                f"group's training end; facet_cutoffs is not written.")
        elif item.facets_fixed and item.facet_cutoffs:
            self.cutoffs_edit.setText(
                ", ".join(str(c) for c in item.facet_cutoffs))
            self.cutoffs_edit.setToolTip(
                f"'{item.display_name}' fixes its facet cutoffs.")
        elif not self.cutoffs_edit.text().strip() and item.facet_cutoffs:
            self.cutoffs_edit.setText(
                ", ".join(str(c) for c in item.facet_cutoffs))
        derived = bool(getattr(item, "data_derived_facets", False))
        ## The phase names of a data-derived type are its own (Training,
        ## Test) and not a config key: shown, never editable, never written.
        self.labels_edit.setEnabled(not derived)
        if derived:
            self.labels_edit.setText(", ".join(item.phase_labels))
            self.labels_edit.setToolTip(
                f"'{item.display_name}' names its phases itself; "
                f"facet_labels is not written.")
        elif not self.labels_edit.text().strip() and item.phase_labels:
            self.labels_edit.setText(", ".join(item.phase_labels))
        for key, value in (item.default_constants or {}).items():
            edit = self.constant_edits.get(key)
            if edit is not None and not edit.text().strip():
                edit.setText(f"{value:g}" if isinstance(value, float)
                             else str(value))
        self._on_layout_changed()

    def _on_layout_changed(self) -> None:
        two_well = self._chamber_layout() == "two_well"
        self._form.setRowVisible(self._well_a_row, two_well)
        self._form.setRowVisible(self._well_b_row, two_well)
        self.params_form.set_chamber_size(2 if two_well else 1)

    def _browse(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Choose the folder to initialize" if self._initialize
            else "Choose (or create) the Project directory",
            self.dir_edit.text() or os.getcwd())
        if chosen:
            self.dir_edit.setText(chosen)

    def _resolve_existing_directory(self, directory: str) -> str | None:
        """The folder to initialize, or None after explaining why not.

        The three ways into a Project stay disjoint: this one is for a folder
        that exists and has no project.yaml.  A missing folder is Create
        Project's job; one that already has a project.yaml is Open a
        Project's.
        """
        path = os.path.abspath(os.path.expanduser(directory))
        name = os.path.basename(os.path.normpath(path))
        if not os.path.isdir(path):
            self._warn(
                f"'{directory}' is not a folder on disk.\n\nInitializing "
                "works on a folder you already have; use 'Create Project…' to "
                "make a new one.")
            return None
        if project_mod.is_project_dir(path):
            QMessageBox.information(
                self, self.windowTitle(),
                f"'{name}' already has a {project_mod.PROJECT_FILENAME} — it "
                "is a Project already.\n\nUse 'Open Project' to work in "
                "it, or 'Project design…' to change its design.")
            return None
        if project_mod.is_experiment_dir(path):
            parent = os.path.dirname(os.path.normpath(path))
            answer = QMessageBox.question(
                self, self.windowTitle(),
                f"'{name}' is an Experiment Directory, not a Project.\n\n"
                f"Initialize '{os.path.basename(parent)}' instead, so "
                f"'{name}' becomes one of its members?")
            if answer != QMessageBox.StandardButton.Yes:
                return None
            if project_mod.is_project_dir(parent):
                QMessageBox.information(
                    self, self.windowTitle(),
                    f"'{os.path.basename(parent)}' is already a Project — use "
                    "'Open Project' to work in it.")
                return None
            ## Show the retarget without re-running the prefill: the design in
            ## the widgets may already have been edited by hand.
            self.dir_edit.blockSignals(True)
            self.dir_edit.setText(parent)
            self.dir_edit.blockSignals(False)
            self.name_edit.setText(os.path.basename(parent))
            path = parent
        return path

    # ------------------------------------------------------------------
    # Loading an existing project.yaml
    # ------------------------------------------------------------------

    def _prefill_from_dir(self) -> None:
        directory = self.dir_edit.text().strip()
        if not directory or not os.path.isdir(directory):
            self.setWindowTitle("Initialize existing directory"
                                if self._initialize else "New Project")
            return
        if project_mod.is_project_dir(directory):
            try:
                with open(os.path.join(directory,
                                       project_mod.PROJECT_FILENAME),
                          encoding="utf-8") as handle:
                    meta = yaml.safe_load(handle) or {}
            except Exception:  # noqa: BLE001 - an unreadable file prefills nothing
                meta = {}
            self.name_edit.setText(str(meta.get("name") or ""))
            self.notes_edit.setPlainText(str(meta.get("notes") or ""))
            design = dict(meta.get("design") or {})
            self._load_design(design.get("global")
                              or self._design_from_members(directory))
            self.setWindowTitle("Edit Project design")
            return
        ## Not a Project yet: a folder of experiments about to become one
        ## still has a design — read it off the first member so the dialog
        ## opens on what is already there rather than on defaults.
        self.setWindowTitle("Initialize existing directory" if self._initialize
                            else "New Project")
        if self._initialize:
            self.name_edit.setText(os.path.basename(
                os.path.normpath(directory)))
        inferred = self._design_from_members(directory)
        if inferred:
            self._load_design(inferred)

    @staticmethod
    def _design_from_members(directory) -> dict:
        """The ``global:`` of the first member found — the migration path for
        a folder of standalone experiments becoming a Project."""
        try:
            for entry in sorted(os.listdir(str(directory))):
                config = os.path.join(str(directory), entry,
                                      project_mod.CONFIG_FILENAME)
                if not os.path.isfile(config):
                    continue
                with open(config, encoding="utf-8") as handle:
                    cfg = yaml.safe_load(handle) or {}
                return dict(cfg.get("global") or {})
        except Exception:  # noqa: BLE001 - inference is a convenience, never a gate
            pass
        return {}

    def _load_design(self, design_global: dict | None) -> None:
        g = dict(design_global or {})
        item = experiment_types.get_experiment_type(g.get("experiment_type"))
        index = self.type_combo.findData(item.name)
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentIndex(max(index, 0))
        self.type_combo.blockSignals(False)

        layout = item.chamber_layout or str(
            g.get("chamber_layout") or "two_well")
        self.layout_combo.blockSignals(True)
        self.layout_combo.setCurrentIndex(
            max(self.layout_combo.findData(layout), 0))
        self.layout_combo.blockSignals(False)

        wells = g.get("well_names") or {}
        self.well_a_edit.setText(str(wells.get("A") or ""))
        self.well_b_edit.setText(str(wells.get("B") or ""))
        self.transform_check.setChecked(bool(g.get("transform_licks", True)))
        cutoffs = g.get("facet_cutoffs") or []
        self.cutoffs_edit.setText(", ".join(str(c) for c in cutoffs))
        self.labels_edit.setText(
            ", ".join(str(v) for v in (g.get("facet_labels") or [])))
        self.exclusion_edit.setText(str(g.get("exclusion_group") or ""))
        self.factors_widget.load_factors(
            g.get("experimental_design_factors") or {})

        chamber_size = 2 if layout == "two_well" else 1
        params = dict(g.get("params") or {})
        params.pop("chamber_size", None)
        self.params_form.load_values(params, chamber_size)
        constants = g.get("constants") or {}
        for key, edit in self.constant_edits.items():
            value = constants.get(key)
            edit.setText("" if value is None else
                         (f"{value:g}" if isinstance(value, float)
                          else str(value)))
        self._on_type_changed()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _build_design(self) -> dict | None:
        """The ``design:`` dict from the widgets, or None after an error."""
        item = self._experiment_type()
        chamber_size = 2 if self._chamber_layout() == "two_well" else 1

        cutoffs = None
        text = self.cutoffs_edit.text().strip()
        if text and not item.facets_fixed:
            try:
                cutoffs = [_number(v) for v in text.split(",") if v.strip()]
            except ValueError:
                self._warn("Facet cutoffs must be numbers, comma-separated.")
                return None

        constants: dict = {}
        for key, label, _placeholder in _CONSTANT_ROWS:
            if chamber_size == 1 and key != "min_untransformed_licks_cutoff":
                continue
            text = self.constant_edits[key].text().strip()
            if not text:
                continue
            try:
                constants[key] = _number(text)
            except ValueError:
                self._warn(f"{label} must be a number.")
                return None

        wells = {}
        if chamber_size == 2:
            for key, edit in (("A", self.well_a_edit), ("B", self.well_b_edit)):
                value = edit.text().strip()
                if value:
                    wells[key] = value

        factors = self.factors_widget.get_factors()
        g = item.build_global(
            params=self.params_form.get_values(),
            well_names=wells,
            constants=constants,
            factors=factors,
            facet_cutoffs=cutoffs,
            transform_licks=self.transform_check.isChecked(),
        )
        if item.chamber_layout is None:
            g["chamber_layout"] = self._chamber_layout()
        labels = [v.strip() for v in self.labels_edit.text().split(",")
                  if v.strip()]
        if getattr(item, "data_derived_facets", False):
            ## No cutoffs exist to count against; the names are the type's.
            labels = []
        if labels:
            n_cutoffs = len(g.get("facet_cutoffs")
                            or item.facet_cutoffs or [])
            if len(labels) != n_cutoffs + 1:
                self._warn(
                    f"{n_cutoffs} cutoff(s) create {n_cutoffs + 1} phases — "
                    f"give {n_cutoffs + 1} names or none.")
                return None
            g["facet_labels"] = labels
        group = self.exclusion_edit.text().strip()
        if group:
            g["exclusion_group"] = group

        problems = item.validate(g)
        if problems:
            self._warn("This design is not valid:\n\n  • "
                       + "\n  • ".join(problems))
            return None
        return {"global": g}

    def _save(self) -> None:
        directory = self.dir_edit.text().strip()
        if not directory:
            self._warn("Choose the folder to initialize." if self._initialize
                       else "Choose the Project directory.")
            return
        if self._initialize:
            resolved = self._resolve_existing_directory(directory)
            if resolved is None:
                return
            directory = resolved
        elif project_mod.is_experiment_dir(directory):
            ## A project.yaml beside a flic_config.yaml makes a Project whose
            ## only member would be its own root — zero members and nothing to
            ## pool.  The Project belongs on the parent.
            self._warn(
                f"'{os.path.basename(directory)}' is an Experiment Directory, "
                "not a Project.  Choose its parent folder instead — the "
                "experiment then becomes one of the Project's members.")
            return
        design = self._build_design()
        if design is None:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            project_mod.create_project_file(
                directory,
                ## In place: the folder names the Project, even when the
                ## confirmation above moved the target up to the parent.
                os.path.basename(os.path.normpath(directory))
                if self._initialize
                else self.name_edit.text().strip() or None,
                self.notes_edit.toPlainText().strip(),
                design=design)
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, self.windowTitle(),
                                 f"Could not write project.yaml:\n{err}")
            return
        self._offer_reinforcement(directory)
        self.saved_dir = directory
        self.accept()

    def _offer_reinforcement(self, directory: str) -> None:
        """Offer to make existing Members inherit the Design just written.

        A Member that carries its own ``global:`` either duplicates the Design
        (harmless but a second place to edit) or contradicts it (the Project
        then refuses to load).  Deleting the block fixes both, and it is the
        only edit: ``dfms:`` and ``scripts:`` are written back untouched.
        """
        try:
            stating = project_mod.members_stating_global(directory)
        except Exception:  # noqa: BLE001 - never block the save on this
            return
        if not stating:
            return
        deviating = [name for name, agrees in stating if not agrees]
        detail = "\n".join(
            f"  • {name}" + ("" if agrees else "  — contradicts the design")
            for name, agrees in stating)
        message = (
            f"{len(stating)} member(s) carry a global: block of their own:\n\n"
            f"{detail}\n\n"
            "Remove those blocks so they inherit this design?  Their dfms: "
            "and scripts: are left untouched.")
        if deviating:
            message += (f"\n\n{len(deviating)} of them contradict the design "
                        "and will refuse to load until this is settled — "
                        "their own values are discarded.")
        answer = QMessageBox.question(
            self, "Members state their own global:", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes if deviating
            else QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.adopted = project_mod.adopt_design(directory)
        except Exception as err:  # noqa: BLE001
            QMessageBox.warning(self, self.windowTitle(),
                                f"Could not update the members:\n{err}")

    def _warn(self, message: str) -> None:
        QMessageBox.warning(self, self.windowTitle(), message)


def open_design_editor(parent: QWidget | None, start_dir) -> str | None:
    """Run the dialog on *start_dir*; return the saved directory, or None."""
    dialog = ProjectDesignDialog(parent, start_dir=start_dir)
    if dialog.exec() and dialog.saved_dir:
        return dialog.saved_dir
    return None

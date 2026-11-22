"""The Project editor: ``project.yaml``'s name, notes and **Design**.

A Project's authority over its Members is only as good as somebody's ability to
state it.  Before this dialog a Project could be created but its ``design:``
section had to be hand-written, so the common path produced a Project with no
Design at all — Members then validated against *each other* (the legacy mode)
and the first one to be edited silently became the standard.

Everything the Design owns (ADR-0005 — experiment type, detection ``params:``,
``well_names``, ``transform_licks``, ``constants:``, the design factors, the
facet cutoffs or ``facet_source``, the exclusion groups and the Slices of
ADR-0020) is authored here, in one place, and every Member inherits it.
The parameter form and the factors and Slices tables are the config editor's
own widgets, so the Design is stated in exactly the vocabulary a Member's
config uses.

Modelled on PyTrackingAnalysis's ``ProjectInfoDialog``.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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

from . import experiment_types, project as project_mod, slices as slices_mod
from .config_editor import (
    _OPTO_CHOICES,
    FactorsWidget,
    ParamsForm,
    SlicesWidget,
    _group_list,
    _group_text,
    _opto_index,
)
from .experiment_types.progressive_ratio import PR_CONSTANT_FIELDS
from .opto_qc import DEFAULT_CONSTANTS as OPTO_DEFAULTS
from .opto_qc import OPTO_CONSTANT_FIELDS
from .program_facets import FACET_SOURCES, SOURCE_LABELS, is_program_source
from .ui import Category, icon
from .ui.widgets import Card, CardGroup, NoWheelComboBox

#: The three optional auto-removal cutoffs, in the order the config editor
#: shows them.  Unchecked → YAML ``null`` (filter off); checked + blank →
#: inherit the type default; checked + number → override.
_CONSTANT_ROWS: tuple[tuple[str, str, str], ...] = (
    ("min_untransformed_licks_cutoff", "Min Untransformed Licks",
     "type default when checked and blank"),
    ("max_med_duration_cutoff", "Max Median Duration",
     "type default when checked and blank"),
    ("max_events_cutoff", "Max Events",
     "type default when checked and blank"),
    ("max_ceiling_minutes", "Max Minutes at Ceiling",
     "type default when checked and blank"),
)
#: The cutoffs that apply to a single-well layout too.
_EVERY_LAYOUT_KEYS = frozenset({"min_untransformed_licks_cutoff", "max_ceiling_minutes"})

#: Progressive Ratio's own constants, (key, label, tooltip) per row, from the
#: one description the Config Editor and the type's validation also read
#: (``experiment_types.progressive_ratio.PR_CONSTANT_FIELDS``): the switches
#: that decide which failed Chamber Groups auto-removal takes out, the light
#: QC's thresholds (pyflic.base.pr_light_qc) and the breaking point's
#: first-gap rule (pyflic.base.pr_breaking_point, ADR-0014).
_PR_SWITCH_ROWS: tuple[tuple[str, str, str], ...] = tuple(
    (f.key, f.label, f.tooltip) for f in PR_CONSTANT_FIELDS if f.group == "switch")
_PR_NUMBER_ROWS: tuple[tuple[str, str, str], ...] = tuple(
    (f.key, f.label, f.tooltip) for f in PR_CONSTANT_FIELDS if f.group == "light_qc")
_PR_BREAK_ROWS: tuple[tuple[str, str, str], ...] = tuple(
    (f.key, f.label, f.tooltip) for f in PR_CONSTANT_FIELDS if f.group == "break")
_PR_KEYS = {key for key, _l, _t in (*_PR_SWITCH_ROWS, *_PR_NUMBER_ROWS, *_PR_BREAK_ROWS)}
_MANAGED_KEYS = {key for key, _l, _p in _CONSTANT_ROWS} | _PR_KEYS
#: The optogenetic light QC's constants (``opto_qc.OPTO_CONSTANT_FIELDS``),
#: every type's; managed only while their group is shown.
_OPTO_KEYS = {f.key for f in OPTO_CONSTANT_FIELDS}


def _add_help(target: Card | CardGroup, ref: str, tooltip: str) -> None:
    """Put a ``?`` opening *ref* in *target*'s title row, if help is available."""
    try:
        from ..help import HelpButton
    except Exception:  # noqa: BLE001 - help is optional, the editor is not
        return
    target.add_title_widget(HelpButton(ref, target, tooltip=tooltip))


def _with_help(widget: QWidget, ref: str, tooltip: str, *,
               grow: bool = True) -> QWidget:
    """*widget* with a ``?`` opening *ref* right after it, as one row — for a
    form field or a section label — or *widget* alone without help."""
    try:
        from ..help import HelpButton
    except Exception:  # noqa: BLE001 - help is optional, the editor is not
        return widget
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addWidget(widget, 1 if grow else 0)
    row.addWidget(HelpButton(ref, holder, tooltip=tooltip), 0,
                  Qt.AlignmentFlag.AlignVCenter)
    if not grow:
        row.addStretch(1)
    return holder


def _add_section(card: Card, text: str, ref: str, tooltip: str) -> None:
    """A section label in *card* with a ``?`` opening *ref* after it."""
    label = QLabel(text, card)
    label.setObjectName("PyflicSectionDivider")
    card.add_body(_with_help(label, ref, tooltip, grow=False))


def _truthy(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in ("", "0", "false", "no", "off")
    return bool(value)


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
        try:
            from ..help import install_help_shortcut

            install_help_shortcut(self, "concepts-project#the-project-design-dialog")
        except Exception:  # noqa: BLE001 - help is optional, the editor is not
            pass

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        body_layout.addWidget(self._build_identity_card(start_dir))
        body_layout.addWidget(self._build_design_card())
        body_layout.addWidget(self._build_slices_card())
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

        ## A finished path, not every keystroke: typing towards one folder
        ## passes through others, and an intermediate one that is a Project
        ## would reload the whole form over what has been edited.
        self.dir_edit.editingFinished.connect(self._on_dir_edited)
        self._prefill_from_dir()
        self._on_type_changed()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_identity_card(self, start_dir) -> Card:
        card = Card("Project", Category.LOAD, icon_name="project")
        _add_help(card, "concepts-project#the-project-design-dialog",
                  "This dialog, card by card  (F1)")
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
        _add_help(card, "concepts-project#the-design-is-an-authority",
                  "What the design owns, and why a member cannot differ")
        form = QFormLayout()
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        ## The shipped types, the lab library's and the copy beside this
        ## Project (ADR-0019); New type… opens the Experiment Type Editor.
        ## The dialog scrolls: the wheel scrolls it and never changes a choice.
        self.type_combo = NoWheelComboBox()
        self._fill_type_combo()
        self.type_combo.currentIndexChanged.connect(self._on_type_changed)
        self.new_type_btn = QPushButton("New type…")
        self.new_type_btn.setIcon(icon("add", category=Category.LOAD))
        self.new_type_btn.setToolTip(
            "Open the Experiment Type Editor to build a type for your lab; it is "
            "listed here once saved.")
        self.new_type_btn.clicked.connect(lambda _checked=False: self._open_type_editor())
        type_row = QHBoxLayout()
        type_row.setContentsMargins(0, 0, 0, 0)
        type_row.addWidget(self.type_combo, 1)
        type_row.addWidget(self.new_type_btn)
        try:
            from ..help import HelpButton

            type_row.addWidget(HelpButton(
                "concepts-experiment-types",
                tooltip="What an Experiment Type decides, and building your own"))
        except Exception:  # noqa: BLE001 - help is optional, the editor is not
            pass
        type_holder = QWidget()
        type_holder.setLayout(type_row)
        form.addRow("Experiment type:", type_holder)
        ## The copy of a lab type beside project.yaml, when it differs from
        ## the library's master: said here, with the update one click away.
        self.type_drift_label = QLabel("")
        self.type_drift_label.setObjectName("PyflicCardSubtitle")
        self.type_drift_label.setWordWrap(True)
        self.type_drift_btn = QPushButton("Update the copy from the library")
        self.type_drift_btn.setIcon(icon("refresh", category=Category.TOOLS))
        self.type_drift_btn.clicked.connect(lambda _checked=False: self.update_type_copy())
        drift_row = QHBoxLayout()
        drift_row.setContentsMargins(0, 0, 0, 0)
        drift_row.addWidget(self.type_drift_label, 1)
        drift_row.addWidget(self.type_drift_btn, 0, Qt.AlignmentFlag.AlignTop)
        self.type_drift_holder = QWidget()
        self.type_drift_holder.setLayout(drift_row)
        self.type_drift_holder.setVisible(False)
        form.addRow("", self.type_drift_holder)

        self.layout_combo = NoWheelComboBox()
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

        self.opto_combo = NoWheelComboBox()
        for label, value in _OPTO_CHOICES:
            self.opto_combo.addItem(label, value)
        self.opto_combo.setToolTip(
            "optogenetics — auto (the default) runs the optogenetic light QC on "
            "every member DFM with a section in its data/Program.txt or a lit LED; "
            "yes on every DFM; no never.  A member's DFM entry may override it for "
            "that DFM, because whether a lid was lit is a fact about the recording.")
        self.opto_combo.currentIndexChanged.connect(self._on_opto_changed)
        form.addRow("Optogenetics:", self.opto_combo)

        ## Where the Facets come from (decision 7): minute cutoffs, or each
        ## DFM's Program Intervals from its Program.txt.  Beside the cutoffs it
        ## replaces; a type that owns its Facets owns this too.
        self.facet_source_combo = NoWheelComboBox()
        for value in FACET_SOURCES:
            self.facet_source_combo.addItem(SOURCE_LABELS[value], value)
        self.facet_source_combo.setToolTip(
            "facet_source — minute cutoffs (the default) split every DFM at the "
            "same minutes; Program Intervals give one Facet per Program Interval "
            "of each DFM's Program.txt, labelled by its Paradigm (Cycles pooled, "
            "or one per Cycle).  A Program Facet needs a Program.txt section for "
            "every DFM, and no cutoffs.")
        self.facet_source_combo.currentIndexChanged.connect(self._on_facet_source_changed)
        form.addRow("Facet source:", _with_help(
            self.facet_source_combo, "concepts-facets",
            "Facets: minute cutoffs and their names, or Program Intervals"))
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
            "The whole recording's own group: the column every member's "
            "remove_chambers.csv is read under and written to, so one rule "
            "filters the whole Project.")
        form.addRow("Exclusion group:", _with_help(
            self.exclusion_edit, "config-structure#globalexclusion-group",
            "The remove_chambers.csv group every member is read under"))
        ## Beside the group it adds to (ADR-0020): an analysis writes only to
        ## its own group, and excludes these as well without inheriting them.
        self.also_exclude_edit = QLineEdit()
        self.also_exclude_edit.setPlaceholderText("e.g. dead  (comma-separated)")
        self.also_exclude_edit.setToolTip(
            "also_exclude — further groups whose chambers the whole recording also "
            "leaves out; it never writes to them.  A fly dead from the start, listed "
            "once in such a group, is left out of every analysis that names it.")
        form.addRow("Also exclude:", _with_help(
            self.also_exclude_edit, "concepts-slices#exclusions",
            "Each analysis's own group, and the groups it also excludes"))

        ## The Treatment control's level (ADR-0019), for a type that compares
        ## every level of a factor with a control: the type names the factor,
        ## the design names the level, since levels change from study to study.
        self.control_level_edit = QLineEdit()
        self.control_level_edit.setToolTip(
            "control_level — the level of the type's control factor every other "
            "level is compared with.")
        self._control_level_row = form.rowCount()
        form.addRow("Control level:", _with_help(
            self.control_level_edit, "concepts-experiment-types#control-comparisons",
            "The Treatment control every other level is compared with"))

        self._form = form
        card.add_body(form)

        _add_section(card, "Design factors", "config-factors",
                     "Factorial designs: factors, levels and assignments")
        hint = QLabel(
            "The factors a chamber assignment names, in this order.  Members "
            "assign levels per chamber; the factors themselves are fixed here.")
        hint.setObjectName("PyflicCardSubtitle")
        hint.setWordWrap(True)
        card.add_body(hint)
        self.factors_widget = FactorsWidget()
        card.add_body(self.factors_widget)
        return card

    def _build_slices_card(self) -> Card:
        ## Declared in the design, so every member runs every Slice its
        ## recording covers and the Combined Analysis pools each by name.
        card = Card("Slices", Category.ANALYZE,
                    subtitle="Optional.  Separate analyses of part of every member's "
                             "recording, each with its own exclusion groups, QC and "
                             "results; the whole recording is always analysed as well.")
        _add_help(card, "concepts-slices#slices-in-the-design",
                  "Slices: what each column means, and how a Slice is analysed")
        self.slices_widget = SlicesWidget()
        card.add_body(self.slices_widget)
        return card

    def _build_params_card(self) -> Card:
        card = Card("Detection parameters", Category.ANALYZE,
                    subtitle="One detection rule across every member — that is "
                            "what makes the Combined Analysis comparable.")
        _add_help(card, "reference-parameters",
                  "Every detection parameter, and which way it moves results")
        self.params_form = ParamsForm(override_mode=False, chamber_size=2,
                                      num_columns=2)
        card.add_body(self.params_form)

        _add_section(card, "Auto-filter thresholds (used by auto_remove_chambers)",
                     "config-dfms-chambers#automatic-exclusion",
                     "What each auto-removal cutoff removes, and the Use box")
        cform = QFormLayout()
        cform.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.constant_checks: dict[str, QCheckBox] = {}
        self.constant_edits: dict[str, QLineEdit] = {}
        for key, label, placeholder in _CONSTANT_ROWS:
            check = QCheckBox("Use")
            check.setToolTip(
                "When checked, auto-removal applies this cutoff (blank value "
                "= type default).  When unchecked, this filter is off.")
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.setMaximumWidth(200)
            edit.setEnabled(False)
            check.toggled.connect(edit.setEnabled)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)
            row_layout.addWidget(check)
            row_layout.addWidget(edit, 1)
            self.constant_checks[key] = check
            self.constant_edits[key] = edit
            cform.addRow(f"{label}:", row)
        card.add_body(cform)

        ## Shown only for a type that carries these constants (Progressive
        ## Ratio); a hidden section writes nothing.
        self.pr_qc_group = CardGroup(
            "Progressive ratio light QC",
            note="Did the paired fly earn its light?  Hover a field for what "
                 "it does.")
        pform = QFormLayout()
        pform.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.pr_switch_checks: dict[str, QCheckBox] = {}
        for key, label, tip in _PR_SWITCH_ROWS:
            check = QCheckBox(label)
            check.setChecked(True)
            check.setToolTip(tip)
            self.pr_switch_checks[key] = check
            pform.addRow("", check)
        self.pr_number_edits: dict[str, QLineEdit] = {}
        for key, label, tip in _PR_NUMBER_ROWS:
            edit = QLineEdit()
            edit.setMaximumWidth(240)
            edit.setToolTip(tip)
            self.pr_number_edits[key] = edit
            pform.addRow(f"{label}:", edit)
        self.pr_qc_group.add(pform)
        _add_help(self.pr_qc_group, "concepts-progressive-ratio#light-qc",
                  "What the light QC checks, and what its thresholds do")
        self.pr_qc_group.setVisible(False)
        card.add_body(self.pr_qc_group)

        ## The breaking point's settings, shown with the light QC's; their
        ## fields share pr_number_edits so loading and saving treat them alike.
        self.pr_break_group = CardGroup(
            "Progressive ratio breaking point",
            note="When did the paired fly stop working for the light?  Hover "
                 "a field for what it does.")
        bform = QFormLayout()
        bform.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for key, label, tip in _PR_BREAK_ROWS:
            edit = QLineEdit()
            edit.setMaximumWidth(240)
            edit.setToolTip(tip)
            if key == "pr_test_window_min":
                edit.setPlaceholderText("blank: no cap")
            self.pr_number_edits[key] = edit
            bform.addRow(f"{label}:", edit)
        self.pr_break_group.add(bform)
        _add_help(self.pr_break_group, "concepts-progressive-ratio#breaking-point",
                  "How the breaking point and sucrose persistence are measured")
        self.pr_break_group.setVisible(False)
        card.add_body(self.pr_break_group)

        ## The optogenetic light QC's thresholds, for every type; hidden when
        ## optogenetics is "no", and then it writes nothing of its own.
        self.opto_group = CardGroup(
            "Optogenetic light QC",
            note="Was the light where the licks were?  Every experiment type; "
                 "blank takes the default.  Hover a field for what it does.")
        oform = QFormLayout()
        oform.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.opto_switch_checks: dict[str, QCheckBox] = {}
        self.opto_number_edits: dict[str, QLineEdit] = {}
        for field in OPTO_CONSTANT_FIELDS:
            default = OPTO_DEFAULTS.get(field.key)
            if field.group == "switch":
                check = QCheckBox(field.label)
                check.setChecked(bool(default))
                check.setToolTip(field.tooltip)
                self.opto_switch_checks[field.key] = check
                oform.addRow("", check)
            else:
                edit = QLineEdit()
                edit.setMaximumWidth(240)
                edit.setToolTip(field.tooltip)
                edit.setPlaceholderText(f"{default:g}  (default)")
                self.opto_number_edits[field.key] = edit
                oform.addRow(f"{field.label}:", edit)
        self.opto_group.add(oform)
        _add_help(self.opto_group, "concepts-optogenetics#thresholds",
                  "What the optogenetic light QC checks, and what its thresholds do")
        card.add_body(self.opto_group)
        return card

    def _on_opto_changed(self) -> None:
        shown = self.opto_combo.currentData() is not False
        self.opto_group.setVisible(shown)
        self.slices_widget.set_opto_columns_visible(shown)

    # ------------------------------------------------------------------
    # Reacting to choices
    # ------------------------------------------------------------------

    def _where(self) -> str | None:
        """The directory being edited, when it exists: a lab type is read
        from the copy beside its project.yaml first (ADR-0019)."""
        edit = getattr(self, "dir_edit", None)
        directory = edit.text().strip() if edit is not None else ""
        return directory if directory and os.path.isdir(directory) else None

    def _fill_type_combo(self, select: str | None = None) -> None:
        """List the types a design here can name, keeping *select* (or the
        current choice) selected when it is still there."""
        current = select or self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        for item in experiment_types.available_experiment_types(self._where()):
            label = item.display_name if item.is_shipped else f"{item.display_name}  (lab)"
            self.type_combo.addItem(label, item.name)
        index = self.type_combo.findData(current) if current else -1
        self.type_combo.setCurrentIndex(max(index, 0))
        self.type_combo.blockSignals(False)

    def _reload_types(self, select: str | None = None) -> None:
        """Re-read the type list (the Experiment Type Editor saved or deleted
        one) and refresh what the chosen type decides."""
        self._fill_type_combo(select or None)
        self._on_type_changed()

    def _open_type_editor(self):
        """The Experiment Type Editor on a new type; a type it saves is
        listed here and chosen."""
        from .type_editor import open_type_editor

        editor = open_type_editor(self)
        editor.typesChanged.connect(self._reload_types)
        editor.closed.connect(lambda: self._reload_types())
        return editor

    def _experiment_type(self):
        try:
            return experiment_types.get_experiment_type(
                self.type_combo.currentData(), self._where())
        except ValueError:
            return experiment_types.get_experiment_type(None)

    def _refresh_type_drift(self) -> None:
        """Show whether the copy beside project.yaml differs from the library's."""
        from .experiment_types import library

        where = self._where()
        drift = library.copy_drift(where) if where else None
        self.type_drift_label.setText(drift or "")
        self.type_drift_holder.setVisible(bool(drift))

    def update_type_copy(self) -> bool:
        """Replace the copy beside project.yaml with the library's master of
        its name; ``False`` when there is nothing to update."""
        from .experiment_types import library

        where = self._where()
        if not where:
            return False
        try:
            copy = library.read_copy(where)
        except ValueError as err:
            self._warn(str(err))
            return False
        master = library.library_definition(copy.name) if copy is not None else None
        if master is None:
            return False
        library.write_copy(master, where)
        self._reload_types(copy.name)
        self._refresh_type_drift()
        return True

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
            "A Custom Experiment states its own layout." if item.is_custom else
            f"'{item.display_name}' leaves the layout to the design.")

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
        self._apply_facet_widgets()
        ## Like the cutoffs above: a type whose Facets come from the data
        ## (the Ratio analysis) has no Slices, and says why.
        self.slices_widget.set_refusal(
            None if slices_mod.allows_slices(item) else slices_mod.refusal(item))
        factor = (item.definition.treatment_control
                  if item.definition is not None else None)
        self._form.setRowVisible(self._control_level_row, factor is not None)
        if factor is not None:
            self.control_level_edit.setPlaceholderText(f"a level of {factor}")
        defaults = item.default_constants or {}
        for key, _label, placeholder in _CONSTANT_ROWS:
            check = self.constant_checks[key]
            edit = self.constant_edits[key]
            value = defaults.get(key)
            if value is None:
                edit.setPlaceholderText(
                    "no default — check and type a number to filter")
            else:
                edit.setPlaceholderText(
                    f"{value:g}  (default for {item.display_name})"
                    if isinstance(value, float)
                    else f"{value}  (default for {item.display_name})")
            ## Do not materialise the type default into the field (ADR-0011).
            ## A blank checked field inherits; only an empty Custom field
            ## follows the type by unchecking.
            if not edit.text().strip():
                loaded = getattr(self, "_loaded_constants", {})
                if key in loaded:
                    enabled = loaded[key] is not None
                else:
                    enabled = value is not None
                check.blockSignals(True)
                check.setChecked(enabled)
                check.blockSignals(False)
                edit.setEnabled(enabled)
        for key, value in defaults.items():
            if key in self.constant_edits:
                continue
            edit = self.pr_number_edits.get(key)
            if edit is not None and not edit.text().strip():
                edit.setText(f"{value:g}" if isinstance(value, float)
                             else str(value))
        pr_type = bool(_PR_KEYS & set(defaults))
        self.pr_qc_group.setVisible(pr_type)
        self.pr_break_group.setVisible(pr_type)
        self._on_layout_changed()

    def _on_facet_source_changed(self) -> None:
        self._apply_facet_widgets()

    def _apply_facet_widgets(self) -> None:
        """The Facet source picker and the two fields it governs: a type that
        owns its Facets fixes the source at cutoffs; Program Facets take
        neither cutoffs nor phase names (each Facet is named by its Paradigm)."""
        item = self._experiment_type()
        owned = item.owns_facet_source
        if owned:
            self.facet_source_combo.blockSignals(True)
            self.facet_source_combo.setCurrentIndex(
                max(self.facet_source_combo.findData("cutoffs"), 0))
            self.facet_source_combo.blockSignals(False)
        self.facet_source_combo.setEnabled(not owned)
        derived = bool(getattr(item, "data_derived_facets", False))
        program = is_program_source(self.facet_source_combo.currentData())
        self.cutoffs_edit.setEnabled(not item.facets_fixed and not program)
        self.labels_edit.setEnabled(not derived and not program)
        if program:
            self.cutoffs_edit.setPlaceholderText(
                "not used: each DFM's Program Intervals are its Facets")
            self.cutoffs_edit.setToolTip(
                "facet_source takes the Facets from Program.txt; facet_cutoffs "
                "is not written.")
            self.labels_edit.setPlaceholderText("not used: each Facet is named by its Paradigm")
        elif not derived:
            self.cutoffs_edit.setPlaceholderText("e.g. 10, 70  (minutes)")
            self.labels_edit.setPlaceholderText("e.g. Acclimation, Experiment, Cooldown")

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
            self._on_dir_edited()

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
            self._prefilled_dir = parent
            self.name_edit.setText(os.path.basename(parent))
            path = parent
        return path

    # ------------------------------------------------------------------
    # Loading an existing project.yaml
    # ------------------------------------------------------------------

    def _on_dir_edited(self) -> None:
        """A directory typed to the end or browsed to: fill the form from it,
        unless it is the one the form was filled from — leaving the field
        after only looking at it must not throw away the edits."""
        if self.dir_edit.text().strip() != self._prefilled_dir:
            self._prefill_from_dir()

    def _prefill_from_dir(self) -> None:
        directory = self.dir_edit.text().strip()
        self._prefilled_dir = directory
        ## What the folder read below states and the form has no field for:
        ## none of the last folder's may reach this one's project.yaml.
        self._loaded_omit = None                # the Selection
        self._loaded_constants = {}             # hand-written constants
        ## A Project's lab type may live only in its copy: list it.
        before = self.type_combo.currentData()
        self._fill_type_combo()
        self._refresh_type_drift()
        if self.type_combo.currentData() != before:
            self._on_type_changed()
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
        """The first member's ``global:``, its factors widened to every level
        any member declares — the migration path for a folder of standalone
        experiments becoming a Project (:func:`project.design_from_members`)."""
        try:
            return project_mod.design_from_members(directory)
        except Exception:  # noqa: BLE001 - inference is a convenience, never a gate
            return {}

    def _load_design(self, design_global: dict | None) -> None:
        g = dict(design_global or {})
        try:
            item = experiment_types.get_experiment_type(g.get("experiment_type"),
                                                        self._where())
        except ValueError:
            item = experiment_types.get_experiment_type(None)
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
        source = item.resolve_facet_source(g)
        self.facet_source_combo.blockSignals(True)
        self.facet_source_combo.setCurrentIndex(
            max(self.facet_source_combo.findData(source), 0))
        self.facet_source_combo.blockSignals(False)
        self.labels_edit.setText(
            ", ".join(str(v) for v in (g.get("facet_labels") or [])))
        self.exclusion_edit.setText(str(g.get("exclusion_group") or ""))
        self.also_exclude_edit.setText(_group_text(g.get("also_exclude")))
        self.slices_widget.load_values(g)
        self.control_level_edit.setText(str(g.get("control_level") or ""))
        self.factors_widget.load_factors(
            g.get("experimental_design_factors") or {})

        chamber_size = 2 if layout == "two_well" else 1
        params = dict(g.get("params") or {})
        params.pop("chamber_size", None)
        self.params_form.load_values(params, chamber_size)
        constants = g.get("constants") or {}
        self._loaded_constants = dict(constants)
        ## The Selection (ADR-0018) has no field here: the Hub's ticks write it.
        self._loaded_omit = g.get("omit")
        defaults = self._experiment_type().default_constants or {}
        for key, edit in self.constant_edits.items():
            check = self.constant_checks[key]
            if key in constants:
                value = constants[key]
                enabled = value is not None
                text = ("" if value is None else
                        (f"{value:g}" if isinstance(value, float)
                         else str(value)))
            else:
                enabled = defaults.get(key) is not None
                text = ""
            check.blockSignals(True)
            check.setChecked(enabled)
            check.blockSignals(False)
            edit.setText(text)
            edit.setEnabled(enabled)
        for key, edit in self.pr_number_edits.items():
            value = constants.get(key)
            edit.setText("" if value is None else
                         (f"{value:g}" if isinstance(value, float)
                          else str(value)))
        for key, check in self.pr_switch_checks.items():
            check.setChecked(_truthy(constants.get(key, True)))
        for key, edit in self.opto_number_edits.items():
            value = constants.get(key)
            edit.setText("" if value is None else
                         (f"{value:g}" if isinstance(value, float) else str(value)))
        for key, check in self.opto_switch_checks.items():
            check.setChecked(_truthy(constants.get(key, OPTO_DEFAULTS.get(key, False))))
        self.opto_combo.blockSignals(True)
        self.opto_combo.setCurrentIndex(_opto_index(g.get("optogenetics")))
        self.opto_combo.blockSignals(False)
        self._on_opto_changed()
        self._on_type_changed()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _build_design(self) -> dict | None:
        """The ``design:`` dict from the widgets, or None after an error."""
        item = self._experiment_type()
        chamber_size = 2 if self._chamber_layout() == "two_well" else 1

        facet_source = ("cutoffs" if item.owns_facet_source
                        else self.facet_source_combo.currentData() or "cutoffs")
        from_program = is_program_source(facet_source)
        cutoffs = None
        text = self.cutoffs_edit.text().strip()
        if text and not item.facets_fixed and not from_program:
            try:
                cutoffs = [_number(v) for v in text.split(",") if v.strip()]
            except ValueError:
                self._warn("Facet cutoffs must be numbers, comma-separated.")
                return None

        constants: dict = {}
        ## Unchecked is null only for a cutoff the type has a default for
        ## (ADR-0011), or one the file already stated: without a default the
        ## filter is off anyway, and a look-and-save must add nothing.
        type_defaults = item.default_constants or {}
        for key, label, _placeholder in _CONSTANT_ROWS:
            if chamber_size == 1 and key not in _EVERY_LAYOUT_KEYS:
                continue
            if not self.constant_checks[key].isChecked():
                if (type_defaults.get(key) is not None
                        or key in (self._loaded_constants or {})):
                    constants[key] = None
                continue
            text = self.constant_edits[key].text().strip()
            if not text:
                continue
            try:
                constants[key] = _number(text)
            except ValueError:
                self._warn(f"{label} must be a number.")
                return None
        if _PR_KEYS & set(item.default_constants or {}):
            for key, check in self.pr_switch_checks.items():
                constants[key] = check.isChecked()
            for key, label, _tip in (*_PR_NUMBER_ROWS, *_PR_BREAK_ROWS):
                text = self.pr_number_edits[key].text().strip()
                if not text:
                    continue
                try:
                    constants[key] = _number(text)
                except ValueError:
                    self._warn(f"{label} must be a number.")
                    return None
        opto_on = self.opto_combo.currentData() is not False
        if opto_on:
            for key, check in self.opto_switch_checks.items():
                ## Written only when it differs from the default, so the
                ## design states a choice rather than today's default.
                if check.isChecked() != bool(OPTO_DEFAULTS.get(key, False)):
                    constants[key] = check.isChecked()
            for field in OPTO_CONSTANT_FIELDS:
                if field.group == "switch":
                    continue
                text = self.opto_number_edits[field.key].text().strip()
                if not text:
                    continue
                try:
                    constants[field.key] = _number(text)
                except ValueError:
                    self._warn(f"{field.label} must be a number.")
                    return None
        ## A constant this form has no field for was written by hand; keep it.
        managed = _MANAGED_KEYS | (_OPTO_KEYS if opto_on else set())
        for key, value in getattr(self, "_loaded_constants", {}).items():
            if key not in managed:
                constants.setdefault(key, value)

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
            facet_source=facet_source,
        )
        if item.chamber_layout is None:
            g["chamber_layout"] = self._chamber_layout()
        labels = [v.strip() for v in self.labels_edit.text().split(",")
                  if v.strip()]
        if getattr(item, "data_derived_facets", False) or from_program:
            ## No cutoffs exist to count against; the names are the type's,
            ## or for Program Facets each Paradigm's.
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
        others = _group_list(self.also_exclude_edit.text())
        if others:
            g["also_exclude"] = others
        declared, slice_problems = self.slices_widget.get_values()
        g.update(declared)
        level = self.control_level_edit.text().strip()
        if level and item.definition is not None and item.definition.treatment_control:
            g["control_level"] = level
        opto = self.opto_combo.currentData()
        if isinstance(opto, bool):
            g["optogenetics"] = opto
        ## Kept as project.yaml says, or a save would undo every untick —
        ## with a renamed Slice's entry renamed, so its box stays unticked.
        omit = self.slices_widget.follow_renames(getattr(self, "_loaded_omit", None))
        if omit:
            g["omit"] = omit

        ## A Slice that cannot be used is skipped by every run, so the design
        ## is refused rather than saved with it: the Project's own check.
        problems = (item.validate(g) + slice_problems
                    + slices_mod.slice_problems(g, item))
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
        new_project = not project_mod.is_project_dir(directory)
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
        self._write_type_copy(directory, new_project=new_project)
        self._offer_reinforcement(directory)
        self.saved_dir = directory
        self.accept()

    def _write_type_copy(self, directory: str, *, new_project: bool) -> None:
        """Carry the design's type with the data (ADR-0019): a lab type is
        copied beside project.yaml, a shipped one leaves no copy behind, and
        a new Project starts with the type's own Plot Specs."""
        from .experiment_types import library

        try:
            item = experiment_types.get_experiment_type(
                self.type_combo.currentData(), directory)
            if item.is_shipped:
                library.remove_copy(directory)
            else:
                library.write_copy(item.definition, directory)
        except (OSError, ValueError) as err:
            self._warn(f"Could not write the experiment type's copy:\n{err}")
            return
        specs = item.plot_specs()
        if not new_project or not specs:
            return
        from . import pubfigures

        if os.path.isfile(pubfigures.specs_path(directory)):
            return
        seeded = pubfigures.ProjectSpecs()
        seeded.plots = {plot_id: pubfigures.PlotSpec.from_dict(spec)
                        for plot_id, spec in specs.items()
                        if plot_id in pubfigures.PLOT_TYPES}
        if seeded.plots:
            pubfigures.save_project_specs(directory, seeded)

    def _offer_reinforcement(self, directory: str) -> None:
        """Offer to write the Design Copy into the Members (ADR-0025).

        Each Member's ``global:`` should hold the Design's description of the
        recording — type, layout, well names, factors, control level — so its
        config reads on its own, and nothing else, so it inherits every
        analysis setting.  Writing it is the only edit: ``dfms:``, ``about:``,
        ``scripts:`` and ``data_folder:`` are written back untouched.

        A Member whose description conflicts with the Design (other well
        names, another layout, a level the Design does not declare) is never
        rewritten: its config says what its recording was.  It is named, with
        the reason, and will not load until a person corrects the Design or
        the Member.
        """
        try:
            states = project_mod.member_design_states(directory)
        except Exception:  # noqa: BLE001 - never block the save on this
            return
        to_write = [s for s in states if s.state == "write"]
        conflicts = [s for s in states if s.state == "conflict"]
        if not to_write and not conflicts:
            return
        conflict_text = ""
        if conflicts:
            conflict_text = (
                f"{len(conflicts)} member(s) describe their recording differently "
                "from this design, so they are left as they are and will not load "
                "until the design or the member is corrected:\n\n"
                + "\n".join(f"  • {s.name} — {'; '.join(s.reasons)}" for s in conflicts))
        if not to_write:
            QMessageBox.warning(self, "Members conflict with the design", conflict_text)
            return
        detail = "\n".join(f"  • {s.name} — {'; '.join(s.reasons)}" for s in to_write)
        message = (
            f"{len(to_write)} member(s) do not carry this design's description "
            f"of the recording:\n\n{detail}\n\n"
            "Write it into their flic_config.yaml, so each reads on its own?  "
            "Any analysis setting a member states is removed (the design owns "
            "those); its dfms:, about:, scripts: and data_folder: are untouched.")
        if conflict_text:
            message += "\n\n" + conflict_text
        answer = QMessageBox.question(
            self, "Write the design into the members", message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes)
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.adopted = project_mod.adopt_design(
                directory, [s.name for s in to_write])
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

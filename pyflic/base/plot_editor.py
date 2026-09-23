"""The Plot Editor (``pyflic-plots``) — a **Project-level** tool.

Opens a Project, renders a live preview of the pooled figures from the same
Spec + Style that saving uses, and writes the vector Publication Figures into
``<project>/figures/``.

Presentation only: it never alters a ``flic_config.yaml``.  Opening a Member
redirects up to its Project, because a Publication Figure is a statement about
the pooled result, not about one recording.

One plot at a time.  The Plot picker in the toolbar chooses which figure the
editor is editing, and the preview shows that one — the same shape as
PyTrackingAnalysis's editor, because only one figure can be previewed at once
and a multi-select list said otherwise.  Every plot the Project can draw is
always part of its figure set; ``plot_specs.yaml`` records how each one is
drawn, not which ones exist.

Everything that shapes the current figure is on one panel, in four groups —
the shared Style, this plot, its Facets, its Treatments — the same column
PyTrackingAnalysis's editor uses.  Tabs hid half the controls behind a click
and made "which of these does the preview answer to" a question; a figure is
one thing and its knobs belong in one place, scrolled rather than paged.

Two spec families share that panel.  The Facets group and the binning row
swap places depending on which family the selected plot belongs to; the
Style group is identical for both, because a Plot Style is what makes a
Project's figures look like one set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import project as project_mod
from . import pubfigures
from .gui_env import sanitize_input_method_environment
from .ui import Category, apply_theme, icon
from .ui import settings as ui_settings
from .ui.widgets import ActionButton, Card, CardGroup


def _readable_on(colour: str) -> str:
    """Black or white, whichever can be read on *colour*.

    A colour button labelled with its own hex has to stay legible against
    every swatch a user picks, including the dark end of the palette.
    """
    c = QColor(colour)
    luminance = (0.299 * c.red() + 0.587 * c.green() + 0.114 * c.blue()) / 255
    return "#000000" if luminance > 0.6 else "#ffffff"


class PlotEditorWindow(QMainWindow):
    """The Plot Editor window."""

    def __init__(self, target: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("pyflic Plot Editor")
        self.resize(1300, 880)

        self.project: project_mod.Project | None = None
        self.specs = pubfigures.ProjectSpecs()
        self.specs.ensure_default_style()
        self._facet_frame = None
        self._binned_frame = None
        self._loading = False

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top = QHBoxLayout()
        open_btn = ActionButton("Open Project…", Category.LOAD, "open",
                                primary=True)
        open_btn.clicked.connect(self._choose_project)
        top.addWidget(open_btn)
        self.project_label = QLabel("No project open")
        top.addWidget(self.project_label, 1)
        top.addWidget(QLabel("Plot:"))
        ## One plot is edited at a time, so the picker is a combo, not a list
        ## with check boxes: the preview can only ever show one figure, and
        ## ticking several implied otherwise.  Nothing here includes or
        ## excludes a figure from the Project's set — every plot the data
        ## supports is rendered; this only chooses which one to work on.
        self.plot_combo = QComboBox()
        self.plot_combo.setMinimumWidth(240)
        self.plot_combo.currentIndexChanged.connect(self._on_plot_selected)
        top.addWidget(self.plot_combo)
        self.reset_btn = ActionButton("Restore defaults", Category.NEUTRAL,
                                      "clear")
        self.reset_btn.setToolTip(
            "Discard this plot's saved Spec and start from the default. "
            "Shared Styles are untouched.")
        self.reset_btn.clicked.connect(self._restore_defaults)
        top.addWidget(self.reset_btn)
        self.save_btn = ActionButton("Save plot_specs.yaml", Category.LOAD,
                                     "save")
        self.save_btn.clicked.connect(self._save_specs)
        top.addWidget(self.save_btn)
        self.render_btn = ActionButton("Render figures", Category.PLOTS, "plot",
                                       primary=True)
        self.render_btn.clicked.connect(self._render)
        top.addWidget(self.render_btn)
        root.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_preview())
        splitter.setSizes([470, 830])
        root.addWidget(splitter, 1)

        if target:
            self.open_target(target)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_left(self) -> QWidget:
        """One scrolling column of groups, not a tab stack.

        Every control here shapes the one figure in the preview, so paging
        half of them behind a tab only asked which half was in force.  The
        groups are ordered as upstream's are: the shared look first, then
        this plot, then the two things it is drawn over.
        """
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        column = QWidget()
        self._controls_lay = QVBoxLayout(column)
        self._controls_lay.setContentsMargins(2, 2, 2, 2)
        self._controls_lay.setSpacing(10)
        self._controls_lay.addWidget(self._build_style_group())
        self._controls_lay.addWidget(self._build_plot_group())
        self._controls_lay.addWidget(self._build_facets_group())
        self._controls_lay.addWidget(self._build_treatments_group())
        self._controls_lay.addStretch(1)

        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        ## Wide enough for the longest field row; the splitter may grow this
        ## side but not squeeze it until the spin boxes clip.
        self.controls_scroll.setMinimumWidth(430)
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        ## The column is a fixed width so a long treatment name can never
        ## push the preview off the window; it scrolls vertically only.
        self.controls_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setWidget(column)
        lay.addWidget(self.controls_scroll, 1)
        return host

    def _build_plot_group(self) -> CardGroup:
        group = CardGroup("This plot")
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        ## A narrow column drops the label onto its own line rather than
        ## shaving the field until its digits are cut off.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self.title_edit = QLineEdit()
        self.title_edit.editingFinished.connect(self._apply_content)
        form.addRow("Title:", self.title_edit)

        self.xlabel_edit = QLineEdit()
        self.xlabel_edit.editingFinished.connect(self._apply_content)
        form.addRow("X label:", self.xlabel_edit)

        self.ylabel_edit = QLineEdit()
        self.ylabel_edit.editingFinished.connect(self._apply_content)
        form.addRow("Y label:", self.ylabel_edit)

        limits = QHBoxLayout()
        self.ymin = QDoubleSpinBox()
        self.ymin.setRange(-1e6, 1e6)
        self.ymax = QDoubleSpinBox()
        self.ymax.setRange(-1e6, 1e6)
        self.use_limits = QCheckBox("Fix y limits")
        for widget in (self.ymin, self.ymax):
            widget.valueChanged.connect(self._apply_content)
        self.use_limits.toggled.connect(self._apply_content)
        limits.addWidget(self.use_limits)
        limits.addWidget(self.ymin)
        limits.addWidget(self.ymax)
        form.addRow("Y range:", limits)

        self.free_y = QCheckBox("Independent y per facet")
        self.free_y.toggled.connect(self._apply_content)
        form.addRow("", self.free_y)

        self.mark_experiments = QCheckBox("Mark members by point shape")
        self.mark_experiments.toggled.connect(self._apply_content)
        form.addRow("", self.mark_experiments)

        ## Family-specific rows, built once and shown or hidden, so switching
        ## plots never rebuilds the form (and never steals focus mid-edit).
        self.binsize = QDoubleSpinBox()
        self.binsize.setRange(0.5, 600.0)
        self.binsize.setValue(30.0)
        self.binsize.valueChanged.connect(self._apply_content)
        self.binsize_label = QLabel("Bin size (min):")
        form.addRow(self.binsize_label, self.binsize)

        self.ribbon = QCheckBox("SEM ribbon")
        self.ribbon.setChecked(True)
        self.ribbon.toggled.connect(self._apply_content)
        self.ribbon_label = QLabel("")
        form.addRow(self.ribbon_label, self.ribbon)

        group.add(page)
        return group

    def _build_facets_group(self) -> CardGroup:
        """The Facets box, hidden whole for a time course — which has none."""
        self.facets_group = CardGroup("Facets")
        self.facet_list = QListWidget()
        self.facet_list.setMaximumHeight(110)
        self.facet_list.itemChanged.connect(self._apply_content)
        self.facets_group.add(self.facet_list)
        return self.facets_group

    def _build_treatments_group(self) -> CardGroup:
        """One row per treatment: shown or not, what it is called, its colour.

        Three decisions about one thing, so three columns of one table — a
        second list keyed by the same names could only ever say the same
        names twice.  The colour sits in the Style, not the Spec, so it is
        marked as shared where it is edited.
        """
        group = CardGroup("Treatments")
        self.treatment_table = QTableWidget(0, 3)
        self.treatment_table.setHorizontalHeaderLabels(
            ["Treatment", "Label", "Colour"])
        self.treatment_table.verticalHeader().setVisible(False)
        header = self.treatment_table.horizontalHeader()
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.treatment_table.setMaximumHeight(150)
        self.treatment_table.itemChanged.connect(self._apply_content)
        group.add(self.treatment_table)
        group.add_note("Untick to leave a treatment out; the Label is what "
                       "the figure prints.  Colours live in the Style, so "
                       "every figure sharing it changes together.")
        return group

    def _build_style_group(self) -> CardGroup:
        group = CardGroup("Style (shared across plots)")
        page = QWidget()
        form = QFormLayout(page)
        form.setContentsMargins(0, 0, 0, 0)
        ## A narrow column drops the label onto its own line rather than
        ## shaving the field until its digits are cut off.
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        row = QHBoxLayout()
        self.style_combo = QComboBox()
        self.style_combo.currentTextChanged.connect(self._on_style_selected)
        row.addWidget(self.style_combo, 1)
        add = QPushButton("New")
        add.clicked.connect(self._new_style)
        row.addWidget(add)
        form.addRow("Style:", row)

        self.width_mm = QDoubleSpinBox()
        self.width_mm.setRange(20.0, 600.0)
        self.height_mm = QDoubleSpinBox()
        self.height_mm.setRange(20.0, 600.0)
        size = QHBoxLayout()
        size.setSpacing(4)
        size.addWidget(self.width_mm, 1)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height_mm, 1)
        form.addRow("Figure (mm):", size)

        ## Sizing per panel rather than per figure, so a plot with three
        ## facets is not three squeezed panels in the width of a one-facet
        ## plot.  Zero means off — the figure keeps the size above, which is
        ## what ``effective_width_mm``/``effective_height_mm`` already do.
        self.facet_width_mm = QDoubleSpinBox()
        self.facet_width_mm.setRange(0.0, 300.0)
        self.facet_width_mm.setSpecialValueText("off")
        form.addRow("Facet width (mm):", self.facet_width_mm)

        self.facet_height_mm = QDoubleSpinBox()
        self.facet_height_mm.setRange(0.0, 300.0)
        self.facet_height_mm.setSpecialValueText("off")
        form.addRow("Facet height (mm):", self.facet_height_mm)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(list(pubfigures._THEMES))
        form.addRow("Theme:", self.theme_combo)

        self.font_edit = QLineEdit()
        form.addRow("Font family:", self.font_edit)

        self.base_pt = QDoubleSpinBox()
        self.base_pt.setRange(4.0, 32.0)
        form.addRow("Base size (pt):", self.base_pt)

        self.geom_combo = QComboBox()
        self.geom_combo.addItems(list(pubfigures._GEOMS))
        form.addRow("Data geometry:", self.geom_combo)

        self.mean_combo = QComboBox()
        self.mean_combo.addItems(list(pubfigures._MEAN_STYLES))
        form.addRow("Mean overlay:", self.mean_combo)

        self.strip_combo = QComboBox()
        self.strip_combo.addItems(list(pubfigures._STRIP_STYLES))
        form.addRow("Facet strips:", self.strip_combo)

        self.point_size = QDoubleSpinBox()
        self.point_size.setRange(0.1, 12.0)
        form.addRow("Point size:", self.point_size)

        self.line_pt = QDoubleSpinBox()
        self.line_pt.setRange(0.1, 6.0)
        form.addRow("Line weight:", self.line_pt)

        for widget in (self.width_mm, self.height_mm, self.facet_width_mm,
                       self.facet_height_mm, self.base_pt,
                       self.point_size, self.line_pt):
            widget.valueChanged.connect(self._apply_style)
        for widget in (self.theme_combo, self.geom_combo, self.mean_combo,
                       self.strip_combo):
            widget.currentTextChanged.connect(self._apply_style)
        self.font_edit.editingFinished.connect(self._apply_style)

        group.add(page)
        return group

    def _build_preview(self) -> QWidget:
        card = Card("Preview", Category.PLOTS, icon_name="plot",
                    subtitle="Rendered by the same Spec + Style that saving "
                             "uses — what you see is what lands in figures/.")
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        ## The figure is fitted to this viewport, so it never needs sideways
        ## scrolling — a figure running off both edges shows the middle of
        ## itself and hides the axes, which is where the reading happens.
        self.preview_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview_scroll.viewport().installEventFilter(self)
        self.preview_label = QLabel("Open a Project to begin.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        card.add_body(self.preview_scroll)
        return card

    def eventFilter(self, watched, event):  # noqa: N802 (Qt override)
        """Refit the preview when its pane changes width.

        Dragging the splitter resizes the viewport without resizing the
        window, so a window-level ``resizeEvent`` would miss it.
        """
        if (watched is self.preview_scroll.viewport()
                and event.type() == QEvent.Type.Resize):
            self._fit_preview()
        return super().eventFilter(watched, event)

    def _fit_preview(self) -> None:
        """Show the rendered figure scaled to the width of its pane.

        Never scaled *up*: the preview claims to be what lands in figures/,
        and a magnified one would promise detail the file does not have.
        """
        source = getattr(self, "_preview_pixmap", None)
        if source is None or source.isNull():
            return
        available = self.preview_scroll.viewport().width() - 8
        if available <= 0:
            return
        if source.width() <= available:
            self.preview_label.setPixmap(source)
            return
        self.preview_label.setPixmap(source.scaledToWidth(
            available, Qt.TransformationMode.SmoothTransformation))

    # ------------------------------------------------------------------
    # Project handling
    # ------------------------------------------------------------------

    def open_target(self, path: str) -> None:
        """Open *path* as a Project, redirecting up from a Member."""
        path = str(Path(path).expanduser().resolve())
        if not project_mod.is_project_dir(path):
            parent = os.path.dirname(path)
            if project_mod.is_experiment_dir(path) and \
                    project_mod.is_project_dir(parent):
                ## A Publication Figure is a statement about the pooled result,
                ## so a member is never the right level to edit one at.
                path = parent
            else:
                QMessageBox.warning(
                    self, "Not a Project",
                    f"{path} has no project.yaml. The Plot Editor works at the "
                    f"Project level.")
                return
        try:
            self.project = project_mod.Project(path)
        except Exception as err:  # noqa: BLE001
            QMessageBox.critical(self, "Could not open", str(err))
            return
        self.specs = pubfigures.load_project_specs(path)
        self._facet_frame, self._binned_frame = pubfigures.project_frames(
            self.project)
        self.project_label.setText(
            f"{self.project.name} — {len(self.project.member_names)} "
            f"member(s), {self.project.chamber_layout}")
        self._reload_lists()
        ## Baseline for the close-time save, taken through the same
        ## normalization the close performs (widget round-trip + pruning the
        ## specs that selection lazily created for unchecked plots): closing
        ## an editor nobody changed must not rewrite the file.
        self._apply_content()
        self._apply_style()
        self._opened_payload = self._specs_payload()

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Project")
        if path:
            self.open_target(path)

    def _reload_lists(self) -> None:
        self._loading = True
        self.plot_combo.clear()
        layout = self.project.chamber_layout if self.project else "two_well"
        type_name = (self.project.experiment_type.name if self.project else None)
        for plot_id in pubfigures.plots_for_layout(layout, type_name):
            self.plot_combo.addItem(pubfigures.PLOT_TYPES[plot_id]["display"],
                                    plot_id)
        self.style_combo.clear()
        self.style_combo.addItems(list(self.specs.styles))
        self.style_combo.setCurrentText(self.specs.default_style)
        ## The combo was filled with ``_loading`` set, so the handler that
        ## fills the Style tab never ran — and the Style tab is what
        ## ``_apply_style`` harvests the style FROM.  Harvesting it empty
        ## overwrote the Project's style with the spin boxes' minimums: a
        ## 20x20 mm figure, which previews as a thumbnail and saves over
        ## plot_specs.yaml on close.  Load it here, before anyone reads it.
        self._load_style_into_form(self.style_combo.currentText())
        self._loading = False
        if self.plot_combo.count():
            self.plot_combo.setCurrentIndex(0)
            self._on_plot_selected()

    # ------------------------------------------------------------------
    # Selection / spec editing
    # ------------------------------------------------------------------

    def current_plot_id(self) -> str | None:
        return self.plot_combo.currentData()

    def current_spec(self) -> pubfigures.PlotSpec | None:
        plot_id = self.current_plot_id()
        if plot_id is None:
            return None
        if plot_id not in self.specs.plots:
            well_a = ((self.project.design_global.get("well_names") or {})
                      .get("A", "well A")) if self.project else "well A"
            self.specs.plots[plot_id] = pubfigures.default_spec(plot_id, well_a)
        return self.specs.plots[plot_id]

    def _on_plot_selected(self, *_args) -> None:
        if self._loading:
            return
        self._load_spec_into_form()
        self._render_preview()

    def _restore_defaults(self) -> None:
        """Throw away this plot's Spec and start from the type's default.

        Content only: the Styles are shared, and resetting one figure must
        not repaint the rest of the set.
        """
        plot_id = self.current_plot_id()
        if plot_id is None or self.project is None:
            return
        self.specs.plots.pop(plot_id, None)
        self.current_spec().style = self.specs.default_style
        self._load_spec_into_form()
        self._render_preview()

    def _load_spec_into_form(self) -> None:
        plot_id = self.current_plot_id()
        spec = self.current_spec()
        if spec is None:
            return
        self._loading = True
        family = pubfigures.family_of(plot_id)
        is_timecourse = family == pubfigures.FAMILY_TIMECOURSE

        self.title_edit.setText(spec.title)
        self.xlabel_edit.setText(spec.x_label)
        self.ylabel_edit.setText(spec.y_label)
        self.use_limits.setChecked(spec.y_limits is not None)
        if spec.y_limits:
            self.ymin.setValue(spec.y_limits[0])
            self.ymax.setValue(spec.y_limits[1])
        self.free_y.setChecked(spec.free_y)
        self.mark_experiments.setChecked(spec.mark_experiments)
        self.binsize.setValue(spec.binsize)
        self.ribbon.setChecked(spec.ribbon)

        self.facets_group.setVisible(not is_timecourse)
        for widget in (self.binsize, self.binsize_label, self.ribbon,
                       self.ribbon_label):
            widget.setVisible(is_timecourse)
        self.free_y.setVisible(not is_timecourse)
        ## Member shapes encode a per-point identity; a time course plots
        ## treatment means, so there is no point to give a shape.
        self.mark_experiments.setVisible(not is_timecourse)

        data = self._data_for(plot_id)
        self.facet_list.clear()
        if not is_timecourse and data is not None and not data.empty:
            phases = list(dict.fromkeys(data["Phase"].astype(str)))
            for phase in phases:
                item = QListWidgetItem(phase)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                included = not spec.facets or phase in spec.facets
                item.setCheckState(Qt.CheckState.Checked if included
                                   else Qt.CheckState.Unchecked)
                self.facet_list.addItem(item)

        self._fill_treatment_table(spec, data)
        self._loading = False

    def _fill_treatment_table(self, spec, data) -> None:
        """Rebuild the Treatments table from *spec* over the data's groups."""
        self.treatment_table.setRowCount(0)
        if data is None or data.empty:
            return
        merged = pubfigures.merged_treatments(spec, data)
        style = self.specs.style_for(spec)
        self.treatment_table.setRowCount(len(merged))
        for row, (name, entry) in enumerate(merged.items()):
            shown = QTableWidgetItem(str(name))
            shown.setData(Qt.ItemDataRole.UserRole, name)
            ## The original name identifies the treatment everywhere — in the
            ## Style's colours and in the data — so it is shown, not edited.
            shown.setFlags((shown.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                           & ~Qt.ItemFlag.ItemIsEditable)
            shown.setCheckState(Qt.CheckState.Checked
                                if entry.get("show", True)
                                else Qt.CheckState.Unchecked)
            self.treatment_table.setItem(row, 0, shown)

            label = QTableWidgetItem(str(entry.get("label", name)))
            self.treatment_table.setItem(row, 1, label)

            colour = style.color_for(name, row)
            button = QPushButton(colour)
            button.setFlat(True)
            button.setStyleSheet(
                f"QPushButton {{ background: {colour}; "
                f"color: {_readable_on(colour)}; border: none; }}")
            button.clicked.connect(
                lambda _c=False, n=name: self._pick_color(n))
            self.treatment_table.setCellWidget(row, 2, button)
        self.treatment_table.resizeColumnToContents(0)

    def _data_for(self, plot_id: str | None):
        if plot_id is None or self.project is None:
            return None
        info = pubfigures.PLOT_TYPES[plot_id]
        if info["family"] == pubfigures.FAMILY_TIMECOURSE:
            source = pubfigures.frame_for(plot_id, self._facet_frame,
                                          self._binned_frame, self.project)
            if source is None or source.empty:
                return None
            return pubfigures.timecourse_data(source, info["metric"])
        source = self._facet_frame
        if source is None or source.empty:
            return None
        order = None
        if "Facet" in source.columns:
            order = list(dict.fromkeys(source["Facet"].astype(str)))
        return pubfigures.faceted_data(source, info["metric"], order)

    def _apply_content(self, *_args) -> None:
        if self._loading:
            return
        spec = self.current_spec()
        if spec is None:
            return
        spec.title = self.title_edit.text()
        spec.x_label = self.xlabel_edit.text()
        spec.y_label = self.ylabel_edit.text()
        spec.y_limits = ([self.ymin.value(), self.ymax.value()]
                         if self.use_limits.isChecked() else None)
        spec.free_y = self.free_y.isChecked()
        spec.mark_experiments = self.mark_experiments.isChecked()
        spec.binsize = self.binsize.value()
        spec.ribbon = self.ribbon.isChecked()

        facets = [self.facet_list.item(i).text()
                  for i in range(self.facet_list.count())
                  if self.facet_list.item(i).checkState() == Qt.CheckState.Checked]
        spec.facets = facets or None

        treatments = {}
        for row in range(self.treatment_table.rowCount()):
            shown = self.treatment_table.item(row, 0)
            label = self.treatment_table.item(row, 1)
            if shown is None:
                continue
            name = shown.data(Qt.ItemDataRole.UserRole)
            treatments[str(name)] = {
                "label": label.text() if label is not None else str(name),
                "show": shown.checkState() == Qt.CheckState.Checked,
            }
        spec.treatments = treatments
        self._render_preview()

    # ---- style ------------------------------------------------------

    def _load_style_into_form(self, name: str) -> None:
        """Fill the Style tab's widgets from the named style.

        The counterpart of :meth:`_apply_style`, and its precondition: those
        widgets are the only place the style is read back from, so every path
        that changes which style is current has to come through here first.
        """
        style = self.specs.styles.get(name)
        if style is None:
            return
        was_loading = self._loading
        self._loading = True
        self.width_mm.setValue(style.width_mm)
        self.height_mm.setValue(style.height_mm)
        self.facet_width_mm.setValue(style.facet_width_mm)
        self.facet_height_mm.setValue(style.facet_height_mm)
        self.theme_combo.setCurrentText(style.theme)
        self.font_edit.setText(style.font_family)
        self.base_pt.setValue(style.base_pt)
        self.geom_combo.setCurrentText(style.geom)
        self.mean_combo.setCurrentText(style.mean_style)
        self.strip_combo.setCurrentText(style.strip_style)
        self.point_size.setValue(style.point_size)
        self.line_pt.setValue(style.line_pt)
        self._loading = was_loading

    def _on_style_selected(self, name: str) -> None:
        if self._loading or not name or name not in self.specs.styles:
            return
        self._load_style_into_form(name)
        spec = self.current_spec()
        if spec is not None:
            spec.style = name
        self._render_preview()

    def _apply_style(self, *_args) -> None:
        if self._loading:
            return
        name = self.style_combo.currentText()
        style = self.specs.styles.get(name)
        if style is None:
            return
        style.width_mm = self.width_mm.value()
        style.height_mm = self.height_mm.value()
        style.facet_width_mm = self.facet_width_mm.value()
        style.facet_height_mm = self.facet_height_mm.value()
        style.theme = self.theme_combo.currentText()
        style.font_family = self.font_edit.text()
        style.base_pt = self.base_pt.value()
        style.geom = self.geom_combo.currentText()
        style.mean_style = self.mean_combo.currentText()
        style.strip_style = self.strip_combo.currentText()
        style.point_size = self.point_size.value()
        style.line_pt = self.line_pt.value()
        self._render_preview()

    def _new_style(self) -> None:
        base = "style"
        index = 1
        while f"{base}{index}" in self.specs.styles:
            index += 1
        name = f"{base}{index}"
        current = self.specs.styles.get(self.style_combo.currentText())
        self.specs.styles[name] = pubfigures.PlotStyle.from_dict(
            current.to_dict() if current else {})
        self.style_combo.addItem(name)
        self.style_combo.setCurrentText(name)

    def _pick_color(self, name: str) -> None:
        style = self.specs.styles.get(self.style_combo.currentText())
        if style is None:
            return
        chosen = QColorDialog.getColor(
            QColor(style.color_for(str(name), 0)), self,
            f"Colour for {name}")
        if not chosen.isValid():
            return
        style.colors[str(name)] = chosen.name()
        self._load_spec_into_form()
        self._render_preview()

    # ------------------------------------------------------------------
    # Preview / output
    # ------------------------------------------------------------------

    def _render_preview(self) -> None:
        plot_id = self.current_plot_id()
        spec = self.current_spec()
        if plot_id is None or spec is None or self.project is None:
            return
        data = self._data_for(plot_id)
        if data is None or data.empty:
            source = pubfigures.source_of(plot_id)
            self._show_preview_message(
                "No paired − yoked curve data saved yet — run the basic "
                "analysis (or plot_pr_cumulative_diff) in each member."
                if source == "pr_diff" else
                "No binned data saved yet — run a binned CSV in each "
                "member." if source == "binned"
                else "No combined analysis yet — build it from the Hub's "
                     "Project panel.")
            return
        try:
            figure = pubfigures.build_figure(
                plot_id, data, spec, self.specs.style_for(spec))
            png = pubfigures.render_png_bytes(
                figure, self.specs.style_for(spec))
        except Exception as err:  # noqa: BLE001
            self._show_preview_message(f"Preview failed: {err}")
            return
        pixmap = QPixmap()
        pixmap.loadFromData(png)
        ## Keep the full-resolution render and fit a copy of it, so widening
        ## the pane sharpens the preview instead of upscaling a thumbnail.
        self._preview_pixmap = pixmap
        self.preview_label.setText("")
        self._fit_preview()

    def _show_preview_message(self, message: str) -> None:
        """Say why there is no figure, in the space the figure would fill.

        A QLabel carries text or a pixmap, never both, and each setter
        clears the other — so setting the (empty) pixmap after the text
        wiped the message and left the preview blank, which is exactly what
        a missing figure already looks like.  Clear first, then speak.
        """
        self._preview_pixmap = None
        self.preview_label.setPixmap(QPixmap())
        self.preview_label.setText(message)

    def _specs_payload(self) -> str:
        """The state a save would write, as a comparable string."""
        import yaml as _yaml

        return _yaml.safe_dump({
            "default_style": self.specs.default_style,
            "styles": {n: s.to_dict() for n, s in self.specs.styles.items()},
            "plots": {i: p.to_dict() for i, p in self.specs.plots.items()},
        }, sort_keys=True)

    def _save_specs(self) -> None:
        if self.project is None:
            return
        path = pubfigures.save_project_specs(
            self.project.project_directory, self.specs)
        self._opened_payload = self._specs_payload()
        QMessageBox.information(self, "Saved", f"Wrote {path}")

    def _render(self) -> None:
        if self.project is None:
            return
        self._save_specs()
        written = pubfigures.render_all(self.project, log=lambda m: None)
        QMessageBox.information(
            self, "Figures rendered",
            f"Wrote {len(written)} figure(s) into "
            f"{os.path.join(self.project.project_directory, 'figures')}")

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        """Closing flushes the full state — every named style and every
        defined plot's spec — into plot_specs.yaml, so nothing designed in
        the dialog is lost to a forgotten Save (mirrors PyTrackingAnalysis's
        Plot Editor).

        Two subtleties.  Text still sitting in a line edit has not commited —
        ``editingFinished`` fires on focus-out, and closeEvent arrives before
        any — so the widget handlers run first.  And an editor nobody changed
        writes nothing: a save round-trips the file through the spec model,
        which prunes keys it does not know, so a look-and-close must not
        rewrite the yaml.
        """
        if self.project is not None:
            self._apply_content()
            self._apply_style()
            if self._specs_payload() != getattr(self, "_opened_payload", None):
                try:
                    pubfigures.save_project_specs(
                        self.project.project_directory, self.specs)
                except Exception as err:  # noqa: BLE001
                    QMessageBox.warning(
                        self, "Could not save plot_specs.yaml", str(err))
        super().closeEvent(event)


def main() -> None:
    sanitize_input_method_environment()
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Plot Editor")
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    target = sys.argv[1] if len(sys.argv) > 1 else None
    window = PlotEditorWindow(target)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

"""The Plot Editor (``pyflic-plots``) — a **Project-level** tool.

Opens a Project, renders a live preview of the pooled figures from the same
Spec + Style that saving uses, and writes the vector Publication Figures into
``<project>/figures/``.

Presentation only: it never alters a ``flic_config.yaml``.  Opening a Replicate
redirects up to its Project, because a Publication Figure is a statement about
the pooled result, not about one recording.

Two spec families share one editor.  The Content tab swaps its facet controls
for binning controls depending on which family the selected plot belongs to;
the Style tab is identical for both, because a Plot Style is what makes a
Project's figures look like one set.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import project as project_mod
from . import pubfigures
from .ui import Category, apply_theme, icon
from .ui import settings as ui_settings
from .ui.widgets import ActionButton, Card


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
        splitter.setSizes([460, 840])
        root.addWidget(splitter, 1)

        if target:
            self.open_target(target)

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_left(self) -> QWidget:
        host = QWidget()
        lay = QVBoxLayout(host)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        plots_card = Card("Figures", Category.PLOTS, icon_name="plots",
                          subtitle="Tick a plot to include it in this "
                                   "project's figure set.")
        self.plot_list = QListWidget()
        self.plot_list.currentItemChanged.connect(self._on_plot_selected)
        self.plot_list.itemChanged.connect(self._on_plot_toggled)
        self.plot_list.setMinimumHeight(190)
        plots_card.add_body(self.plot_list)
        lay.addWidget(plots_card)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_content_tab(), "Content")
        self.tabs.addTab(self._build_style_tab(), "Style")
        lay.addWidget(self.tabs, 1)
        return host

    def _build_content_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

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

        self.mark_experiments = QCheckBox("Mark replicates by point shape")
        self.mark_experiments.toggled.connect(self._apply_content)
        form.addRow("", self.mark_experiments)

        ## Family-specific rows. Both are built once and shown or hidden, so
        ## switching plots never rebuilds the form (and never loses focus).
        self.facet_list = QListWidget()
        self.facet_list.setMaximumHeight(90)
        self.facet_list.itemChanged.connect(self._apply_content)
        self.facet_row_label = QLabel("Facets:")
        form.addRow(self.facet_row_label, self.facet_list)

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

        self.treatment_list = QListWidget()
        self.treatment_list.setMaximumHeight(110)
        self.treatment_list.itemChanged.connect(self._apply_content)
        form.addRow("Treatments:", self.treatment_list)
        return page

    def _build_style_tab(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)

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
        size.addWidget(self.width_mm)
        size.addWidget(QLabel("×"))
        size.addWidget(self.height_mm)
        form.addRow("Size (mm):", size)

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

        for widget in (self.width_mm, self.height_mm, self.base_pt,
                       self.point_size, self.line_pt):
            widget.valueChanged.connect(self._apply_style)
        for widget in (self.theme_combo, self.geom_combo, self.mean_combo,
                       self.strip_combo):
            widget.currentTextChanged.connect(self._apply_style)
        self.font_edit.editingFinished.connect(self._apply_style)

        self.color_list = QListWidget()
        self.color_list.setMaximumHeight(110)
        self.color_list.itemDoubleClicked.connect(self._pick_color)
        form.addRow("Treatment colors:", self.color_list)
        form.addRow("", QLabel("Double-click a treatment to set its color."))
        return page

    def _build_preview(self) -> QWidget:
        card = Card("Preview", Category.PLOTS, icon_name="plot",
                    subtitle="Rendered by the same Spec + Style that saving "
                             "uses — what you see is what lands in figures/.")
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_label = QLabel("Open a Project to begin.")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_scroll.setWidget(self.preview_label)
        card.add_body(self.preview_scroll)
        return card

    # ------------------------------------------------------------------
    # Project handling
    # ------------------------------------------------------------------

    def open_target(self, path: str) -> None:
        """Open *path* as a Project, redirecting up from a Replicate."""
        path = str(Path(path).expanduser().resolve())
        if not project_mod.is_project_dir(path):
            parent = os.path.dirname(path)
            if project_mod.is_experiment_dir(path) and \
                    project_mod.is_project_dir(parent):
                ## A Publication Figure is a statement about the pooled result,
                ## so a replicate is never the right level to edit one at.
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
            f"{self.project.name} — {len(self.project.experiment_names)} "
            f"replicate(s), {self.project.chamber_layout}")
        self._reload_lists()

    def _choose_project(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a Project")
        if path:
            self.open_target(path)

    def _reload_lists(self) -> None:
        self._loading = True
        self.plot_list.clear()
        layout = self.project.chamber_layout if self.project else "two_well"
        for plot_id in pubfigures.plots_for_layout(layout):
            item = QListWidgetItem(pubfigures.PLOT_TYPES[plot_id]["display"])
            item.setData(Qt.ItemDataRole.UserRole, plot_id)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked
                               if plot_id in self.specs.plots
                               else Qt.CheckState.Unchecked)
            self.plot_list.addItem(item)
        self.style_combo.clear()
        self.style_combo.addItems(list(self.specs.styles))
        self.style_combo.setCurrentText(self.specs.default_style)
        self._loading = False
        if self.plot_list.count():
            self.plot_list.setCurrentRow(0)

    # ------------------------------------------------------------------
    # Selection / spec editing
    # ------------------------------------------------------------------

    def current_plot_id(self) -> str | None:
        item = self.plot_list.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def current_spec(self) -> pubfigures.PlotSpec | None:
        plot_id = self.current_plot_id()
        if plot_id is None:
            return None
        if plot_id not in self.specs.plots:
            well_a = ((self.project.design_global.get("well_names") or {})
                      .get("A", "well A")) if self.project else "well A"
            self.specs.plots[plot_id] = pubfigures.default_spec(plot_id, well_a)
        return self.specs.plots[plot_id]

    def _on_plot_toggled(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        plot_id = item.data(Qt.ItemDataRole.UserRole)
        if item.checkState() == Qt.CheckState.Unchecked:
            self.specs.plots.pop(plot_id, None)
        else:
            self.current_spec()
        self._render_preview()

    def _on_plot_selected(self, *_args) -> None:
        if self._loading:
            return
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

        for widget in (self.facet_list, self.facet_row_label):
            widget.setVisible(not is_timecourse)
        for widget in (self.binsize, self.binsize_label, self.ribbon,
                       self.ribbon_label):
            widget.setVisible(is_timecourse)
        self.free_y.setVisible(not is_timecourse)
        ## Replicate shapes encode a per-point identity; a time course plots
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

        self.treatment_list.clear()
        self.color_list.clear()
        if data is not None and not data.empty:
            merged = pubfigures.merged_treatments(spec, data)
            style = self.specs.style_for(spec)
            for index, (name, entry) in enumerate(merged.items()):
                item = QListWidgetItem(str(entry.get("label", name)))
                item.setData(Qt.ItemDataRole.UserRole, name)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                              | Qt.ItemFlag.ItemIsEditable)
                item.setCheckState(Qt.CheckState.Checked
                                   if entry.get("show", True)
                                   else Qt.CheckState.Unchecked)
                self.treatment_list.addItem(item)

                color = style.color_for(name, index)
                swatch = QListWidgetItem(f"{name}  —  {color}")
                swatch.setData(Qt.ItemDataRole.UserRole, name)
                pixmap = QPixmap(14, 14)
                pixmap.fill(Qt.GlobalColor.transparent)
                from PyQt6.QtGui import QColor, QPainter

                painter = QPainter(pixmap)
                painter.fillRect(0, 0, 14, 14, QColor(color))
                painter.end()
                swatch.setIcon(icon("plot") if False else swatch.icon())
                swatch.setData(Qt.ItemDataRole.DecorationRole, pixmap)
                self.color_list.addItem(swatch)
        self._loading = False

    def _data_for(self, plot_id: str | None):
        if plot_id is None or self.project is None:
            return None
        info = pubfigures.PLOT_TYPES[plot_id]
        if info["family"] == pubfigures.FAMILY_TIMECOURSE:
            source = self._binned_frame
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
        for index in range(self.treatment_list.count()):
            item = self.treatment_list.item(index)
            name = item.data(Qt.ItemDataRole.UserRole)
            treatments[str(name)] = {
                "label": item.text(),
                "show": item.checkState() == Qt.CheckState.Checked,
            }
        spec.treatments = treatments
        self._render_preview()

    # ---- style ------------------------------------------------------

    def _on_style_selected(self, name: str) -> None:
        if self._loading or not name or name not in self.specs.styles:
            return
        style = self.specs.styles[name]
        self._loading = True
        self.width_mm.setValue(style.width_mm)
        self.height_mm.setValue(style.height_mm)
        self.theme_combo.setCurrentText(style.theme)
        self.font_edit.setText(style.font_family)
        self.base_pt.setValue(style.base_pt)
        self.geom_combo.setCurrentText(style.geom)
        self.mean_combo.setCurrentText(style.mean_style)
        self.strip_combo.setCurrentText(style.strip_style)
        self.point_size.setValue(style.point_size)
        self.line_pt.setValue(style.line_pt)
        self._loading = False
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

    def _pick_color(self, item: QListWidgetItem) -> None:
        name = item.data(Qt.ItemDataRole.UserRole)
        style = self.specs.styles.get(self.style_combo.currentText())
        if style is None:
            return
        chosen = QColorDialog.getColor(parent=self)
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
            family = pubfigures.family_of(plot_id)
            self.preview_label.setText(
                "No binned data saved yet — run a binned CSV in each "
                "replicate." if family == pubfigures.FAMILY_TIMECOURSE
                else "No combined analysis yet — build it from the Hub's "
                     "Project panel.")
            self.preview_label.setPixmap(QPixmap())
            return
        try:
            figure = pubfigures.build_figure(
                plot_id, data, spec, self.specs.style_for(spec))
            png = pubfigures.render_png_bytes(
                figure, self.specs.style_for(spec))
        except Exception as err:  # noqa: BLE001
            self.preview_label.setText(f"Preview failed: {err}")
            self.preview_label.setPixmap(QPixmap())
            return
        pixmap = QPixmap()
        pixmap.loadFromData(png)
        self.preview_label.setPixmap(pixmap)
        self.preview_label.setText("")

    def _save_specs(self) -> None:
        if self.project is None:
            return
        path = pubfigures.save_project_specs(
            self.project.project_directory, self.specs)
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


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("pyflic Plot Editor")
    apply_theme(app, mode=ui_settings.get("theme", "auto"))
    target = sys.argv[1] if len(sys.argv) > 1 else None
    window = PlotEditorWindow(target)
    window.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()

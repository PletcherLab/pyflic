"""The help window — one modeless, shared window for every help button.

Qt's ``QTextBrowser.setMarkdown()`` renders GitHub-dialect markdown (tables
included) but emits **no anchors** for headings, so ``scrollToAnchor()`` cannot
be used.  Anchor navigation instead walks the document's blocks looking at
``blockFormat().headingLevel()`` and matches the heading text by slug — see
``docs/adr/0004-help-rendering-and-reference-validation.md``.  Do not "simplify" this back
to ``scrollToAnchor``; it silently does nothing.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QDesktopServices,
    QFont,
    QTextBlockFormat,
    QTextCharFormat,
    QTextCursor,
    QTextDocument,
)
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTextBrowser,
    QToolBar,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..base.ui import settings as ui_settings
from ..base.ui.icons import icon
from . import topics as _topics
from .toc import GUIDES, TOC, guide, guide_neighbours, neighbours

# Tree items carry (topic_id, guide_id | None).
_NAV_ROLE = Qt.ItemDataRole.UserRole

_MISSING_MD = """# Help topic not available

The topic `{topic_id}` could not be loaded.

Nothing else in pyflic is affected — analysis, plotting and scripts work
normally. Use the topic list on the left to find what you need.
"""


class HelpWindow(QMainWindow):
    """Modeless help browser.  Use :func:`pyflic.help.open_help` to reach it."""

    navigated = pyqtSignal(str)

    def __init__(self) -> None:
        # Deliberately parentless: the help window outlives whichever dialog or
        # app opened it, so closing that app's window does not close the help.
        super().__init__(None)
        self.setWindowTitle("pyflic Help")

        # Each entry is (topic_id, anchor, guide_id).
        self._history: list[tuple[str, str | None, str | None]] = []
        self._pos = -1
        self._syncing = False
        self._nav_generation = 0

        self._build_toolbar()
        self._build_body()
        self._populate_tree()
        self._restore_geometry()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = QToolBar("Navigation", self)
        bar.setMovable(False)
        self.addToolBar(bar)

        self._act_back = bar.addAction(icon("fa5s.arrow-left"), "Back")
        self._act_back.setShortcut("Alt+Left")
        self._act_back.triggered.connect(self.go_back)

        self._act_fwd = bar.addAction(icon("fa5s.arrow-right"), "Forward")
        self._act_fwd.setShortcut("Alt+Right")
        self._act_fwd.triggered.connect(self.go_forward)

        bar.addSeparator()
        act_contents = bar.addAction(icon("fa5s.list"), "Contents")
        act_contents.setToolTip("Go to the first topic")
        act_contents.triggered.connect(lambda: self.show_topic(_first_topic_id()))

        spacer = QWidget(self)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        bar.addWidget(spacer)

        self._search = QLineEdit(self)
        self._search.setPlaceholderText("Search help…")
        self._search.setClearButtonEnabled(True)
        self._search.setMaximumWidth(320)
        self._search.textChanged.connect(self._on_search_changed)
        bar.addWidget(self._search)

    def _build_body(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        self._tree = QTreeWidget(splitter)
        self._tree.setHeaderHidden(True)
        self._tree.setMinimumWidth(210)
        self._tree.itemSelectionChanged.connect(self._on_tree_selection)
        splitter.addWidget(self._tree)

        right = QWidget(splitter)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)

        # Stack: 0 = rendered topic, 1 = search results.
        self._stack = QStackedWidget(right)

        self._view = QTextBrowser(self._stack)
        self._view.setOpenLinks(False)          # links are routed by _on_link
        self._view.setOpenExternalLinks(False)
        self._view.anchorClicked.connect(self._on_link)
        self._stack.addWidget(self._view)

        self._results = QListWidget(self._stack)
        self._results.itemActivated.connect(self._on_result_chosen)
        self._results.itemClicked.connect(self._on_result_chosen)
        self._stack.addWidget(self._results)

        right_lay.addWidget(self._stack, 1)

        footer = QWidget(right)
        f_lay = QHBoxLayout(footer)
        f_lay.setContentsMargins(8, 4, 8, 6)
        self._btn_prev = QPushButton("‹ Previous", footer)
        self._btn_next = QPushButton("Next ›", footer)
        self._btn_prev.clicked.connect(lambda: self._step(-1))
        self._btn_next.clicked.connect(lambda: self._step(1))
        self._lbl_where = QLabel("", footer)
        self._lbl_where.setStyleSheet("color: palette(mid);")
        f_lay.addWidget(self._btn_prev)
        f_lay.addStretch(1)
        f_lay.addWidget(self._lbl_where)
        f_lay.addStretch(1)
        f_lay.addWidget(self._btn_next)
        right_lay.addWidget(footer)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([240, 680])
        self.setCentralWidget(splitter)

    def _populate_tree(self) -> None:
        self._tree.clear()

        def add_leaf(parent: QTreeWidgetItem, topic_id: str, guide_id: str | None) -> None:
            topic = _topics.load(topic_id)
            item = QTreeWidgetItem(parent, [topic.title if topic else topic_id])
            item.setData(0, _NAV_ROLE, (topic_id, guide_id))

        guides_root = QTreeWidgetItem(self._tree, ["Guides"])
        _make_header(guides_root)
        for g in GUIDES:
            g_item = QTreeWidgetItem(guides_root, [g.title])
            g_item.setToolTip(0, g.summary)
            g_item.setFlags(g_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for topic_id in g.topic_ids:
                add_leaf(g_item, topic_id, g.id)
        guides_root.setExpanded(True)

        all_root = QTreeWidgetItem(self._tree, ["All topics"])
        _make_header(all_root)
        for section in TOC:
            s_item = QTreeWidgetItem(all_root, [section.title])
            s_item.setFlags(s_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            for topic_id in section.topic_ids:
                add_leaf(s_item, topic_id, None)
            s_item.setExpanded(True)
        all_root.setExpanded(True)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def show_topic(
        self,
        topic_id: str,
        anchor: str | None = None,
        guide_id: str | None = None,
    ) -> None:
        """Render *topic_id*, optionally at *anchor*, and record it in history.

        When *guide_id* is given, Previous/Next follow that guide's reading
        order instead of the overall table of contents.
        """
        entry = (topic_id, anchor, guide_id)
        if 0 <= self._pos < len(self._history) and self._history[self._pos] == entry:
            return
        self._history = self._history[: self._pos + 1]
        self._history.append(entry)
        self._pos = len(self._history) - 1
        self._render(*entry)

    def go_back(self) -> None:
        if self._pos > 0:
            self._pos -= 1
            self._render(*self._history[self._pos])

    def go_forward(self) -> None:
        if self._pos + 1 < len(self._history):
            self._pos += 1
            self._render(*self._history[self._pos])

    def _step(self, direction: int) -> None:
        current = self.current_topic_id()
        if current is None:
            return
        guide_id = self.current_guide_id()
        prev_id, next_id = (
            guide_neighbours(guide_id, current) if guide_id else neighbours(current)
        )
        target = next_id if direction > 0 else prev_id
        if target:
            self.show_topic(target, None, guide_id)

    def current_topic_id(self) -> str | None:
        if 0 <= self._pos < len(self._history):
            return self._history[self._pos][0]
        return None

    def current_guide_id(self) -> str | None:
        if 0 <= self._pos < len(self._history):
            return self._history[self._pos][2]
        return None

    def _render(self, topic_id: str, anchor: str | None, guide_id: str | None) -> None:
        # Bumped on every render so queued work can tell whether it is stale.
        self._nav_generation += 1
        topic = _topics.load(topic_id)
        source = topic.source if topic else _MISSING_MD.format(topic_id=topic_id)

        self._view.document().setMarkdown(
            source, QTextDocument.MarkdownFeature.MarkdownDialectGitHub
        )
        self._style_headings()
        self._stack.setCurrentIndex(0)
        self._view.verticalScrollBar().setValue(0)
        if anchor:
            # Defer until the layout has settled.  ``open_help`` navigates
            # before ``show()``, and a document that has not been laid out
            # yet reports the wrong cursor rect, landing the reader a couple
            # of sections off target.
            #
            # The generation guard matters: navigating away before the timer
            # fires would otherwise scroll the *new* topic to a heading that
            # happens to share the old topic's anchor.
            generation = self._nav_generation
            QTimer.singleShot(
                0, lambda a=anchor, g=generation: self._scroll_if_current(a, g)
            )

        self._sync_tree(topic_id, guide_id)
        self._sync_footer(topic_id, guide_id)
        self._act_back.setEnabled(self._pos > 0)
        self._act_fwd.setEnabled(self._pos + 1 < len(self._history))
        title = topic.title if topic else topic_id
        self.setWindowTitle(f"pyflic Help — {title}")
        self.navigated.emit(topic_id)

    #: Heading level → (point-size multiplier, space above in px).
    _HEADING_STYLE: dict[int, tuple[float, int]] = {
        1: (1.75, 0),
        2: (1.30, 20),
        3: (1.12, 14),
        4: (1.02, 10),
    }

    def _style_headings(self) -> None:
        """Give headings visible weight and breathing room.

        ``setMarkdown`` builds the document directly rather than parsing HTML,
        so ``setDefaultStyleSheet`` never applies to it.  Formatting is applied
        by walking the blocks instead — the same walk anchors use.

        The selection is made explicitly from start-of-block to end-of-block.
        ``SelectionType.BlockUnderCursor`` looks equivalent but also spans the
        *preceding* block separator, which propagates the heading's block
        format onto the paragraph above it — giving ordinary paragraphs a
        non-zero ``headingLevel()`` and corrupting the metadata that
        :meth:`_scroll_to_heading` matches against.
        """
        doc = self._view.document()
        base = self._view.font().pointSizeF()
        if base <= 0:
            base = 10.0

        # Collect first: mutating block formats while iterating the document
        # invalidates the blocks we are walking.
        targets: list[tuple[int, int]] = []
        block = doc.begin()
        while block.isValid():
            level = block.blockFormat().headingLevel()
            if level:
                targets.append((block.position(), level))
            block = block.next()

        if not targets:
            return

        cursor = QTextCursor(doc)
        cursor.beginEditBlock()
        try:
            for position, level in targets:
                scale, space_above = self._HEADING_STYLE.get(level, (1.0, 8))
                char = QTextCharFormat()
                char.setFontPointSize(base * scale)
                char.setFontWeight(QFont.Weight.Bold)

                c = QTextCursor(doc)
                c.setPosition(position)
                blk = QTextBlockFormat(c.blockFormat())
                blk.setTopMargin(space_above)
                blk.setBottomMargin(4)
                c.setBlockFormat(blk)

                c.movePosition(QTextCursor.MoveOperation.StartOfBlock)
                c.movePosition(
                    QTextCursor.MoveOperation.EndOfBlock,
                    QTextCursor.MoveMode.KeepAnchor,
                )
                c.mergeCharFormat(char)
        finally:
            cursor.endEditBlock()

    def _scroll_if_current(self, anchor: str, generation: int) -> bool:
        """Scroll to *anchor* only if no navigation has happened since."""
        if generation != self._nav_generation:
            return False
        return self._scroll_to_heading(anchor)

    def _scroll_to_heading(self, anchor: str) -> bool:
        """Scroll so the heading matching *anchor* sits at the top of the view.

        Matches the slug of the heading text.  ``setMarkdown`` produces no HTML
        anchors, so walking heading blocks is the only way to address a heading.
        """
        want = _topics.slugify(anchor)
        if not want:
            return False
        doc = self._view.document()
        block = doc.begin()
        while block.isValid():
            if block.blockFormat().headingLevel() and _topics.slugify(block.text()) == want:
                cursor = QTextCursor(block)
                self._view.setTextCursor(cursor)
                self._view.ensureCursorVisible()
                sb = self._view.verticalScrollBar()
                sb.setValue(sb.value() + self._view.cursorRect(cursor).top())
                return True
            block = block.next()
        return False

    def _sync_tree(self, topic_id: str, guide_id: str | None) -> None:
        self._syncing = True
        try:
            fallback: QTreeWidgetItem | None = None
            for item in _leaves(self._tree):
                data = item.data(0, _NAV_ROLE)
                if not data or data[0] != topic_id:
                    continue
                if data[1] == guide_id:
                    self._tree.setCurrentItem(item)
                    self._tree.scrollToItem(item)
                    return
                fallback = fallback or item
            if fallback is not None:
                self._tree.setCurrentItem(fallback)
                self._tree.scrollToItem(fallback)
        finally:
            self._syncing = False

    def _sync_footer(self, topic_id: str, guide_id: str | None) -> None:
        if guide_id:
            prev_id, next_id = guide_neighbours(guide_id, topic_id)
            g = guide(guide_id)
            where = g.title if g else ""
            if g and topic_id in g.topic_ids:
                where = f"{g.title}  ·  {g.topic_ids.index(topic_id) + 1} of {len(g.topic_ids)}"
        else:
            prev_id, next_id = neighbours(topic_id)
            section = next((s for s in TOC if topic_id in s.topic_ids), None)
            where = section.title if section else ""

        self._lbl_where.setText(where)
        for btn, tid, fmt in (
            (self._btn_prev, prev_id, "‹ {}"),
            (self._btn_next, next_id, "{} ›"),
        ):
            btn.setEnabled(tid is not None)
            if tid:
                t = _topics.load(tid)
                btn.setText(fmt.format(t.title if t else tid))
            else:
                btn.setText(fmt.format("—"))

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_tree_selection(self) -> None:
        if self._syncing:
            return
        item = self._tree.currentItem()
        if item is None:
            return
        data = item.data(0, _NAV_ROLE)
        if data:
            self.show_topic(data[0], None, data[1])

    def _on_link(self, url: QUrl) -> None:
        """Route markdown links: internal topics navigate, external ones open.

        Links are authored as ``other-topic.md#heading`` so the same markdown
        also resolves correctly when read on GitHub, where topics are siblings.
        """
        if url.scheme() in ("http", "https", "mailto"):
            QDesktopServices.openUrl(url)
            return
        anchor = url.fragment() or None
        path = url.path().lstrip("./")
        if path.endswith(".md"):
            path = path[:-3]
        if not path:
            if anchor:
                self._scroll_to_heading(anchor)
            return
        # Unknown targets still navigate, so the reader gets the graceful
        # "topic not available" page rather than a click that does nothing.
        self.show_topic(path, anchor, self.current_guide_id())

    def _on_search_changed(self, text: str) -> None:
        query = text.strip()
        if len(query) < 2:
            self._stack.setCurrentIndex(0)
            return
        self._results.clear()
        hits = _topics.search(query)
        if not hits:
            item = QListWidgetItem(f"No matches for “{query}”")
            item.setFlags(Qt.ItemFlag.NoItemFlags)
            self._results.addItem(item)
        for hit in hits:
            where = f"{hit.title} › {hit.heading}" if hit.heading else hit.title
            item = QListWidgetItem(f"{where}\n    {hit.snippet}")
            item.setData(_NAV_ROLE, (hit.topic_id, hit.heading))
            self._results.addItem(item)
        self._stack.setCurrentIndex(1)

    def _on_result_chosen(self, item: QListWidgetItem) -> None:
        payload = item.data(_NAV_ROLE)
        if not payload:
            return
        topic_id, heading = payload
        self.show_topic(topic_id, heading)
        self._search.clear()

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _restore_geometry(self) -> None:
        geo = ui_settings.get("help_geometry", None)
        if isinstance(geo, list) and len(geo) == 4:
            try:
                self.setGeometry(*(int(v) for v in geo))
                return
            except (TypeError, ValueError):
                pass
        self.resize(960, 720)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        g = self.geometry()
        ui_settings.set_value("help_geometry", [g.x(), g.y(), g.width(), g.height()])
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_header(item: QTreeWidgetItem) -> None:
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
    font = item.font(0)
    font.setBold(True)
    item.setFont(0, font)


def _leaves(tree: QTreeWidget):
    """Yield every item of *tree*, depth-first."""
    stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
    while stack:
        item = stack.pop(0)
        if item is None:
            continue
        yield item
        stack.extend(item.child(i) for i in range(item.childCount()))


def _first_topic_id() -> str:
    for section in TOC:
        if section.topic_ids:
            return section.topic_ids[0]
    return "getting-started"


# ---------------------------------------------------------------------------
# Single shared instance
# ---------------------------------------------------------------------------

_window: HelpWindow | None = None


def help_window() -> HelpWindow:
    """The one shared help window, created on first use."""
    global _window
    if _window is None or not _is_alive(_window):
        _window = HelpWindow()
    return _window


def _is_alive(win: HelpWindow) -> bool:
    try:
        win.isVisible()
        return True
    except RuntimeError:      # underlying C++ object deleted
        return False


def open_help(
    topic_id: str | None = None,
    anchor: str | None = None,
    guide_id: str | None = None,
) -> HelpWindow | None:
    """Show the help window at *topic_id* (``"id#anchor"`` is accepted).

    Returns ``None`` when there is no ``QApplication`` — help must never be the
    reason a headless run fails.
    """
    if QApplication.instance() is None:
        return None
    if topic_id and anchor is None:
        topic_id, anchor = _topics.parse_ref(topic_id)
    win = help_window()
    win.show_topic(topic_id or _first_topic_id(), anchor, guide_id)
    win.show()
    win.raise_()
    win.activateWindow()
    return win

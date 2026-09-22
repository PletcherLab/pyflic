"""pyflic.base — the analysis library and the apps built on it.

**matplotlib must load before Qt.**  matplotlib's wheel carries its own
FreeType, statically linked into ``matplotlib.ft2font`` with its ``FT_*``
symbols exported; PyQt6's ``libQt6Gui`` loads the system ``libfreetype.so.6``.
Two copies of one library in one process bind each other's calls by load
order, so when Qt loads first, matplotlib's text renderer ends up inside a
FreeType it was not built against and fails while drawing the first tick
label (``FT_Render_Glyph … raster overflow``) — reproducibly, in any thread,
whenever a figure is saved after a ``QApplication`` exists.  Importing
matplotlib here, before any app module can import PyQt6, binds matplotlib to
its own FreeType.  Every entry point (Hub, Config Editor, QC Viewer, Plot
Editor, the CLI) imports this package first, so this one line covers them all.
"""

import matplotlib  # noqa: F401  (see the module docstring — order matters)

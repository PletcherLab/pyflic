"""Graphical Script Editor for pyflic YAML configs.

Public entry points:

* :class:`ScriptEditorWindow` — embeddable ``QMainWindow``
* :func:`launch` — stand-alone runner for development

See the plan at ``/home/scott/.claude/plans/`` for the design rationale.
"""

from .window import ScriptEditorWindow, launch

__all__ = ["ScriptEditorWindow", "launch"]

"""Environment hygiene for the pyflic desktop apps."""

from __future__ import annotations

import os

# GNOME/IBus on Wayland manages these through ibus-ui-gtk3.  Leaving inherited
# module overrides set can make Qt/GTK complain at app startup.
INPUT_METHOD_ENV_VARS = ("QT_IM_MODULE", "GTK_IM_MODULE")


def sanitize_input_method_environment() -> None:
    """Remove desktop-session input-method overrides before Qt starts."""
    for name in INPUT_METHOD_ENV_VARS:
        os.environ.pop(name, None)

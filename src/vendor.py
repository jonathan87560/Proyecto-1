"""Activa las dependencias de terceros incluidas con S08."""

import sys
from pathlib import Path

_VENDOR_DIR = Path(__file__).parents[1] / "_vendor"


def activar_vendor() -> None:
    """Prioriza el código vendorizado sobre una instalación del entorno."""

    ruta = str(_VENDOR_DIR)
    if ruta not in sys.path:
        sys.path.insert(0, ruta)

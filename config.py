"""Runtime bootstrap for AI Masters ML3 notebooks.

Locally, notebooks import helpers from the checked-out repository. In Google
Colab, the seminar-specific config function downloads only the helper modules
needed by that notebook. Package installation belongs to requirements.txt.
"""

from importlib import import_module, invalidate_caches, reload
from pathlib import Path
import sys
from urllib.request import urlopen


GITHUB_REPOSITORY = "SergeyMalashenko/2026_ML3"
DEFAULT_BRANCH = "main"

try:
    import google.colab  # type: ignore[import-not-found]  # noqa: F401

    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False


SEMINAR_FILES = {
    1: (
        "plots/__init__.py",
        "plots/seminar01.py",
        "plots/assets/training_loop.svg",
    ),
}


def _download(relative_path, branch=DEFAULT_BRANCH):
    destination = Path(relative_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/"
        f"{branch}/{relative_path}"
    )
    with urlopen(url, timeout=30) as response:
        destination.write_bytes(response.read())


def config_seminar(number, branch=DEFAULT_BRANCH):
    """Make helper modules for one seminar available in Google Colab."""
    if number not in SEMINAR_FILES:
        raise ValueError(f"No helper manifest for seminar {number}")

    if not IS_COLAB:
        return

    if all(Path(name).is_file() for name in SEMINAR_FILES[number]):
        return

    print(f"Загрузка модулей семинара {number:02d}...")
    for relative_path in SEMINAR_FILES[number]:
        _download(relative_path, branch)
    invalidate_caches()
    print("Модули курса готовы.")


def config_seminar01(branch=DEFAULT_BRANCH):
    """Prepare seminar 01 and return fresh plot helpers, including in a warm kernel."""
    root_path = str(Path(__file__).resolve().parent)
    if root_path in sys.path:
        sys.path.remove(root_path)
    sys.path.insert(0, root_path)
    config_seminar(1, branch)
    invalidate_caches()
    # A previous notebook may have imported a different package named plots.
    reload(import_module("plots"))
    return reload(import_module("plots.seminar01"))

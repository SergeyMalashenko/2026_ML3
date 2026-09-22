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


def select_device(requested=None):
    """Course priority: MPS, CUDA, CPU; explicit requests never fall back."""
    import os
    import torch

    requested = requested or os.environ.get("AI_MASTERS_DEVICE", "auto")
    available = {
        "mps": torch.backends.mps.is_built() and torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
        "cpu": True,
    }
    if requested == "auto":
        requested = next(name for name, enabled in available.items() if enabled)
    if requested not in available or not available[requested]:
        raise RuntimeError(f"Requested device is unavailable: {requested}")
    return torch.device(requested)

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
    2: ("plots/__init__.py", "plots/seminar02.py"),
    3: ("plots/__init__.py", "plots/seminar03.py"),
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


def config_seminar02(branch=DEFAULT_BRANCH):
    """Prepare seminar 02 using the shared course environment."""
    import shutil
    from importlib.util import find_spec

    if IS_COLAB and (find_spec("torchviz") is None or find_spec("graphviz") is None):
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "torchviz>=0.0.3",
                        "graphviz>=0.21"], check=True)
    if shutil.which("dot") is None:
        raise RuntimeError("Нужен Graphviz (команда dot). Установите Graphviz в общее окружение системы.")
    root = Path(__file__).resolve().parent
    if IS_COLAB:
        import hashlib
        import tempfile

        # Do not activate a mixture of cached and newly downloaded modules.
        payload = {}
        for name in SEMINAR_FILES[2]:
            url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{branch}/{name}"
            with urlopen(url, timeout=30) as response:
                payload[name] = response.read()
            if name.endswith(".py"):
                compile(payload[name], name, "exec")
        digest = hashlib.sha256()
        for name, content in sorted(payload.items()):
            digest.update(name.encode() + b"\0" + content + b"\0")
        cache = root / ".course_helpers"
        cache.mkdir(exist_ok=True)
        version = cache / digest.hexdigest()
        if not version.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / "ready"
                for name, content in payload.items():
                    destination = staged / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                staged.rename(version)
        # Detect damaged caches instead of silently running them.
        for name, content in payload.items():
            if (version / name).read_bytes() != content:
                raise RuntimeError("Повреждён кеш модулей курса; удалите .course_helpers и повторите запуск.")
        root = version
    root_path = str(root)
    if root_path in sys.path:
        sys.path.remove(root_path)
    sys.path.insert(0, root_path)
    invalidate_caches()
    for name in list(sys.modules):
        if name == "plots" or name.startswith("plots."):
            del sys.modules[name]
    return import_module("plots.seminar02")


def config_seminar03(branch=DEFAULT_BRANCH):
    """Load Chapter 2 helpers; activate only a complete, validated download."""
    root = Path(__file__).resolve().parent
    if IS_COLAB:
        import hashlib
        import tempfile
        from importlib.util import find_spec

        if find_spec("tensorboard") is None:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install",
                            "tensorboard>=2.16"], check=True)
        payload = {}
        for name in SEMINAR_FILES[3]:
            url = f"https://raw.githubusercontent.com/{GITHUB_REPOSITORY}/{branch}/{name}"
            with urlopen(url, timeout=30) as response:
                payload[name] = response.read()
            compile(payload[name], name, "exec")
        digest = hashlib.sha256()
        for name, content in sorted(payload.items()):
            digest.update(name.encode() + b"\0" + content + b"\0")
        cache = root / ".course_helpers"
        cache.mkdir(exist_ok=True)
        version = cache / digest.hexdigest()
        if not version.exists():
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / "ready"
                for name, content in payload.items():
                    destination = staged / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(content)
                staged.rename(version)
        for name, content in payload.items():
            if (version / name).read_bytes() != content:
                raise RuntimeError("Повреждён кеш модулей курса; удалите .course_helpers и повторите запуск.")
        root = version
    root_path = str(root)
    if root_path in sys.path:
        sys.path.remove(root_path)
    sys.path.insert(0, root_path)
    invalidate_caches()
    for name in list(sys.modules):
        if name == "plots" or name.startswith("plots."):
            del sys.modules[name]
    return import_module("plots.seminar03")

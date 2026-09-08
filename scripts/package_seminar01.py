"""Build the unpublished seminar's Colab helper archive from its manifest."""

from pathlib import Path
import runpy
from zipfile import ZIP_DEFLATED, ZipFile


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = runpy.run_path(str(root / "config.py"))["SEMINAR_FILES"][1]
    destination = root / "notebooks/01_backprop/seminar01_colab_helpers.zip"
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        for relative_path in ("config.py", *manifest):
            archive.write(root / relative_path, arcname=relative_path)
    print(destination.relative_to(root))


if __name__ == "__main__":
    main()

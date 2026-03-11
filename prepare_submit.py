# Creates a zip file for submission on Gradescope.

import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"


def _collect_py_files(dir_path: Path, prefix: str = "") -> list[str]:
    out = []
    for p in dir_path.iterdir():
        rel = f"{prefix}{p.name}" if prefix else p.name
        if p.is_file() and p.suffix == ".py":
            out.append(rel)
        elif p.is_dir() and not p.name.startswith(".") and p.name != "__pycache__":
            out.extend(_collect_py_files(p, prefix=f"{rel}/"))
    return out


def _required_files():
    files = [p for p in os.listdir(REPO_ROOT) if p.endswith(".py") and Path(REPO_ROOT, p).is_file()]
    files.extend(_collect_py_files(SRC, prefix="src/"))
    if (REPO_ROOT / "predictions").exists():
        files.extend(f"predictions/{p}" for p in os.listdir(REPO_ROOT / "predictions"))
    return files


required_files = _required_files()

def main():
    aid = "cs224n_default_final_project_submission"

    with zipfile.ZipFile(f"{aid}.zip", "w") as zz:
        for file in required_files:
            path = REPO_ROOT / file
            if path.exists():
                zz.write(path, file)
    print(f"Submission zip file created: {aid}.zip")

if __name__ == '__main__':
    main()

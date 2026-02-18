#!/usr/bin/env python3

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def repo_root_from_this_file() -> Path:
    # this file: cosmos-supervisor/src/approach1/run_level1_all.py
    # parents[0]=approach1, [1]=src, [2]=cosmos-supervisor (repo root)
    return Path(__file__).resolve().parents[2]


def run_one(root: Path, level1_py: Path, input_mp4: Path, out_jsonl: Path) -> None:
    if not input_mp4.exists():
        raise FileNotFoundError(f"Missing input video: {input_mp4}")

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(level1_py),
        "--input",
        str(input_mp4),
        "--out",
        str(out_jsonl),
    ]

    print("\n" + "=" * 80)
    print(f"INPUT : {input_mp4.relative_to(root)}")
    print(f"OUTPUT: {out_jsonl.relative_to(root)}")
    print("CMD   :", " ".join([
        "python", str(level1_py.relative_to(root)),
        "--input", str(input_mp4.relative_to(root)),
        "--out", str(out_jsonl.relative_to(root)),
    ]))
    print("=" * 80)
    subprocess.run(cmd, check=True)



def main() -> int:
    root = repo_root_from_this_file()

    level1_py = root / "src" / "approach1" / "level1_classify_full_video.py"
    if not level1_py.exists():
        print(f"[ERROR] Missing: {level1_py}")
        return 2

    videos_dir = root / "videos" / "inputs"
    out_dir = root / "outputs" / "approach1"

    jobs = [
        (
            videos_dir / "success" / "success_1.mp4",
            out_dir / "level1_fullvideo_success.jsonl",
        ),
        (
            videos_dir / "obstruction" / "obstruction_1.mp4",
            out_dir / "level1_fullvideo_obstruction.jsonl",
        ),
        (
            videos_dir / "push_down" / "push_down_1.mp4",
            out_dir / "level1_fullvideo_push_down.jsonl",
        ),
    ]

    try:
        for inp, outp in jobs:
            run_one(root=root, level1_py=level1_py, input_mp4=inp, out_jsonl=outp)
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Level-1 run failed with exit code {e.returncode}")
        return e.returncode
    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1

    print("\n✅ Done. Level-1 outputs are in: outputs/approach1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


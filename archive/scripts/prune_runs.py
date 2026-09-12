#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path
from typing import List


def user() -> str:
    return os.environ.get("USER", "unknown")


def default_root() -> Path:
    scratch = Path("/scratch") / user() / "runs"
    if scratch.is_dir():
        return scratch
    return Path("/cache/home") / user() / "runs"


def fmt_size(n: int) -> str:
    units = ["B", "K", "M", "G", "T", "P"]
    x = float(n)
    for unit in units:
        if x < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(x)}{unit}"
            return f"{x:.1f}{unit}"
        x /= 1024.0
    return f"{n}B"


def dir_size(root: Path) -> int:
    total = 0
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            with os.scandir(path) as it:
                for entry in it:
                    try:
                        if entry.is_symlink():
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except FileNotFoundError:
                        continue
        except FileNotFoundError:
            continue
    return total


def kids(root: Path) -> List[Path]:
    out = []
    for path in root.iterdir():
        if path.is_dir():
            out.append(path)
    return out


def norm_target(root: Path, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if root == path or root not in path.parents:
        raise ValueError(f"target must stay under {root}: {raw}")
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="List or delete large run directories.")
    p.add_argument("--root", type=Path, default=default_root(), help="Run root to inspect.")
    p.add_argument("-n", "--top", type=int, default=20, help="How many directories to show.")
    p.add_argument("--min-gb", type=float, default=0.0, help="Only show directories at or above this size.")
    p.add_argument("--rm", nargs="*", default=[], metavar="DIR", help="Delete these directories under --root.")
    p.add_argument("--yes", action="store_true", help="Required with --rm.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"missing root: {root}")

    if args.rm:
        if not args.yes:
            raise SystemExit("--rm requires --yes")
        reclaimed = 0
        for raw in args.rm:
            path = norm_target(root, raw)
            if not path.exists():
                print(f"skip missing {path}")
                continue
            size = dir_size(path)
            shutil.rmtree(path)
            reclaimed += size
            print(f"rm {path} {fmt_size(size)}")
        print(f"reclaimed {fmt_size(reclaimed)}")
        return 0

    rows = []
    floor = int(args.min_gb * (1024 ** 3))
    for path in kids(root):
        size = dir_size(path)
        if size >= floor:
            rows.append((size, path))
    rows.sort(reverse=True)
    for size, path in rows[: max(0, int(args.top))]:
        print(f"{fmt_size(size):>8}  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

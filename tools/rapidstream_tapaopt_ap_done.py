#!/usr/bin/env python3
"""Run RapidStream TAPA optimization with the ap_done FSM patch applied.

This is a thin wrapper around ``rapidstream-tapaopt``.  It preserves the normal
RapidStream command line, but intercepts ``--run-impl`` so the flow becomes:

1. run RapidStream optimization/export without implementation,
2. patch all generated solution XO files in place,
3. run the generated v++ scripts in parallel.

Without ``--run-impl`` the wrapper simply runs RapidStream and patches the
generated solution XO files, so later manual v++ commands can use the fixed XO.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable

from patch_rapidstream_ap_done import patch_xo


def split_wrapper_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--rapidstream-bin", default="rapidstream-tapaopt")
    parser.add_argument("--impl-workers", type=int, default=2)
    parser.add_argument("--patch-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("-h", "--help", action="store_true")
    ns, rest = parser.parse_known_args(argv)
    if ns.help:
        print(__doc__)
        print("\nWrapper options:")
        parser.print_help()
        print("\nRapidStream options:")
        subprocess.run([ns.rapidstream_bin, "--help"], check=False)
        raise SystemExit(0)
    return ns, rest


def pop_flag(args: list[str], flag: str) -> bool:
    found = False
    kept: list[str] = []
    for arg in args:
        if arg == flag:
            found = True
        else:
            kept.append(arg)
    args[:] = kept
    return found


def get_option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for idx, arg in enumerate(args):
        if arg == name and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(prefix):
            return arg[len(prefix) :]
    return None


def solution_dirs(work_dir: Path) -> list[Path]:
    dse_dir = work_dir / "dse"
    if dse_dir.exists():
        return sorted(p for p in dse_dir.glob("solution_*") if p.is_dir())
    return sorted(p for p in work_dir.rglob("solution_*") if p.is_dir())


def candidate_xos(work_dir: Path) -> list[Path]:
    xos: list[Path] = []
    for sol in solution_dirs(work_dir):
        for xo in sol.rglob("*.xo"):
            if xo.name.endswith("_global_done_fix.xo"):
                continue
            if ".temp" in xo.parts or "vitis_run_" in str(xo):
                continue
            xos.append(xo)
    return sorted(set(xos))


def patch_xos_in_place(xos: Iterable[Path], dry_run: bool = False) -> None:
    xos = list(xos)
    if not xos:
        raise RuntimeError("no solution XO files found to patch")
    for xo in xos:
        print(f"[patch] {xo}", flush=True)
        if not dry_run:
            tmp_xo = xo.with_suffix(xo.suffix + ".ap_done_tmp")
            patch_xo(xo, tmp_xo, allow_clean=True)
            tmp_xo.replace(xo)


def vpp_scripts(work_dir: Path) -> list[Path]:
    scripts: list[Path] = []
    for sol in solution_dirs(work_dir):
        for path in sol.rglob("*.sh"):
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            if "v++" in text and ".xo" in text:
                scripts.append(path)
    return sorted(set(scripts))


def run_script(script: Path, dry_run: bool = False) -> int:
    print(f"[v++] {script}", flush=True)
    if dry_run:
        return 0
    with subprocess.Popen(["bash", str(script)], cwd=script.parent) as proc:
        return proc.wait()


def run_vpp_scripts(scripts: list[Path], workers: int, dry_run: bool = False) -> int:
    if not scripts:
        raise RuntimeError("no generated v++ scripts found")
    failed = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        future_to_script = {
            executor.submit(run_script, script, dry_run): script for script in scripts
        }
        for future in as_completed(future_to_script):
            script = future_to_script[future]
            rc = future.result()
            if rc != 0:
                failed += 1
                print(f"[v++] failed rc={rc}: {script}", file=sys.stderr, flush=True)
            else:
                print(f"[v++] completed: {script}", flush=True)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    wrapper, rs_args = split_wrapper_args(argv)
    work_dir_value = get_option_value(rs_args, "--work-dir")
    if not work_dir_value:
        raise RuntimeError("--work-dir is required")
    work_dir = Path(work_dir_value).resolve()

    requested_run_impl = pop_flag(rs_args, "--run-impl")
    rapidstream_bin = shutil.which(wrapper.rapidstream_bin) or wrapper.rapidstream_bin

    if not wrapper.patch_only:
        cmd = [rapidstream_bin, *rs_args]
        print("[rapidstream] " + " ".join(cmd), flush=True)
        if not wrapper.dry_run:
            subprocess.run(cmd, check=True)

    patch_xos_in_place(candidate_xos(work_dir), dry_run=wrapper.dry_run)

    if requested_run_impl:
        return run_vpp_scripts(
            vpp_scripts(work_dir),
            workers=wrapper.impl_workers,
            dry_run=wrapper.dry_run,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

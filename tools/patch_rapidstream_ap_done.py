#!/usr/bin/env python3
"""Patch RapidStream/TAPA global FSM ap_done wiring.

RapidStream-generated XO/HDL can leave ``global_fsm_ap_done`` unassigned in
``__global_fsm_*_fsm.v`` even though ``ap_done`` and ``ap_ready`` are driven by
the internal ``ap_done__q0`` signal.  Vitis can still link such designs, but the
host may observe incorrect completion behavior.  This utility patches the
generated global FSM by adding:

    assign global_fsm_ap_done = ap_done__q0;

The input may be either a RapidStream solution/build directory or a ``.xo``.
For ``.xo`` inputs, the script unpacks, patches, and repacks the archive.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


GLOBAL_FSM_GLOB = "__global_fsm*_fsm.v"
PATCH_LINE = "    assign global_fsm_ap_done = ap_done__q0;\n"


def patch_fsm_file(path: Path) -> bool:
    text = path.read_text()
    if "global_fsm_ap_done" not in text:
        return False
    if "assign global_fsm_ap_done" in text:
        return False
    needle = "    assign ap_ready = ap_done__q0;\n"
    if needle not in text:
        raise RuntimeError(f"{path}: could not find ap_ready assignment")
    text = text.replace(needle, needle + PATCH_LINE, 1)
    path.write_text(text)
    return True


def patch_tree(root: Path) -> list[Path]:
    patched: list[Path] = []
    for path in root.rglob(GLOBAL_FSM_GLOB):
        if patch_fsm_file(path):
            patched.append(path)
    return patched


def unpack_xo(xo: Path, work_dir: Path) -> None:
    shutil.unpack_archive(str(xo), str(work_dir), "zip")


def repack_xo(work_dir: Path, output_xo: Path) -> None:
    output_xo.parent.mkdir(parents=True, exist_ok=True)
    if output_xo.exists():
        output_xo.unlink()
    subprocess.run(
        ["zip", "-qr", str(output_xo), "."],
        cwd=work_dir,
        check=True,
    )


def patch_xo(input_xo: Path, output_xo: Path, allow_clean: bool = False) -> list[Path]:
    with tempfile.TemporaryDirectory(prefix="patch_rapidstream_ap_done_") as td:
        work_dir = Path(td)
        unpack_xo(input_xo, work_dir)
        patched = patch_tree(work_dir)
        if not patched:
            if not allow_clean:
                raise RuntimeError(f"{input_xo}: no global FSM files needed patching")
            output_xo.parent.mkdir(parents=True, exist_ok=True)
            if input_xo.resolve() != output_xo.resolve():
                shutil.copy2(input_xo, output_xo)
            return patched
        repack_xo(work_dir, output_xo)
        return patched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RapidStream solution/build directory or .xo")
    parser.add_argument("-o", "--output", type=Path, help="Output .xo path for .xo input")
    parser.add_argument("--in-place", action="store_true", help="Patch input directory or .xo in place")
    parser.add_argument("--allow-clean", action="store_true", help="Do not fail if the input is already patched")
    args = parser.parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        parser.error(f"input does not exist: {input_path}")

    if input_path.is_dir():
        if args.output:
            parser.error("--output is only valid for .xo input")
        patched = patch_tree(input_path)
        print(f"patched {len(patched)} file(s)")
        for path in patched:
            print(path)
        return 0

    if input_path.suffix != ".xo":
        parser.error("file input must be a .xo archive")

    if args.in_place:
        output_xo = input_path
        tmp_output = input_path.with_suffix(input_path.suffix + ".patched_tmp")
        patched = patch_xo(input_path, tmp_output, allow_clean=args.allow_clean)
        os.replace(tmp_output, input_path)
    else:
        output_xo = args.output
        if output_xo is None:
            output_xo = input_path.with_name(input_path.stem + "_global_done_fix.xo")
        patched = patch_xo(input_path, output_xo.resolve(), allow_clean=args.allow_clean)

    print(f"patched {len(patched)} file(s)")
    print(f"wrote {output_xo}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)

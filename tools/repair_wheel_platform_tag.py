#!/usr/bin/env python3
import argparse
import base64
import csv
import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

GLIBC_VERSION_PATTERN = re.compile(r"GLIBC_(\d+)\.(\d+)")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description=
    "Inspect a wheel's ELF payload and optionally normalize it: rewrite the RPATH of every "
    "extension to a relocatable value and retag the wheel to the manylinux floor its GLIBC "
    "symbols imply.")
  parser.add_argument("wheel", type=Path, help="Path to the wheel file to inspect.")
  parser.add_argument(
    "--retag",
    action="store_true",
    help="Rewrite the wheel filename and WHEEL metadata to match the detected platform tag.",
  )
  parser.add_argument(
    "--set-rpath",
    default=None,
    metavar="RPATH",
    help="Rewrite the RPATH of every extension (pass $ORIGIN) and repack the wheel.  A "
    "wheel built in a conda environment points its extensions at that environment's lib "
    "directory -- a path that is meaningless on a user's machine, and a wrong library if it "
    "happens to exist there.  It is not needed either: the torch libraries an extension "
    "needs are loaded by `import torch` first, so their SONAMEs are already resolved.",
  )
  parser.add_argument(
    "--print-rpath",
    action="store_true",
    help="Append the RPATH of the first extension to the printed line.",
  )
  return parser.parse_args()


def _glibc_version_key(version_text: str) -> tuple[int, int]:
  major_text, minor_text = version_text.split(".", 1)
  return int(major_text), int(minor_text)


def _detect_glibc_versions(elf_path: Path) -> set[str]:
  output = subprocess.check_output(["objdump", "-T", str(elf_path)], text=True)
  return {f"{major}.{minor}" for major, minor in GLIBC_VERSION_PATTERN.findall(output)}


def _detect_wheel_platform_tag(wheel_path: Path) -> str:
  with tempfile.TemporaryDirectory(prefix="cache_dit_wheel_tag_") as temp_dir:
    extract_root = Path(temp_dir)
    glibc_versions: set[str] = set()
    with ZipFile(wheel_path) as wheel_zip:
      for member in wheel_zip.infolist():
        if not member.filename.endswith(".so"):
          continue
        extracted_path = extract_root / member.filename
        extracted_path.parent.mkdir(parents=True, exist_ok=True)
        extracted_path.write_bytes(wheel_zip.read(member.filename))
        glibc_versions.update(_detect_glibc_versions(extracted_path))

  if not glibc_versions:
    return "linux_x86_64"

  max_glibc = max(glibc_versions, key=_glibc_version_key)
  return f"manylinux_2_{max_glibc.split('.', 1)[1]}_x86_64"


def _parse_filename_platform_tag(wheel_path: Path) -> tuple[str, str, str, str]:
  if wheel_path.suffix != ".whl":
    raise ValueError(f"Expected a .whl file, got: {wheel_path}")

  head, python_tag, abi_tag, platform_tag = wheel_path.stem.rsplit("-", 3)
  return head, python_tag, abi_tag, platform_tag


def _hash_file(file_path: Path) -> tuple[str, str]:
  digest = hashlib.sha256(file_path.read_bytes()).digest()
  digest_text = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
  return f"sha256={digest_text}", str(file_path.stat().st_size)


def _rewrite_record(extract_root: Path, dist_info_dir: Path) -> None:
  record_path = dist_info_dir / "RECORD"
  rows: list[list[str]] = []
  for file_path in sorted(path for path in extract_root.rglob("*") if path.is_file()):
    relative_path = file_path.relative_to(extract_root).as_posix()
    if relative_path == f"{dist_info_dir.name}/RECORD":
      rows.append([relative_path, "", ""])
      continue
    digest_text, size_text = _hash_file(file_path)
    rows.append([relative_path, digest_text, size_text])

  with record_path.open("w", newline="") as record_file:
    csv.writer(record_file).writerows(rows)


def _detect_wheel_rpath(wheel_path: Path) -> str:
  """The RPATH of the first extension, which is the one the others carry too."""
  with ZipFile(wheel_path) as wheel_zip:
    members = [
      member.filename for member in wheel_zip.infolist() if member.filename.endswith(".so")
    ]
    if not members:
      return ""
    payload = wheel_zip.read(members[0])

  with tempfile.TemporaryDirectory(prefix="cache_dit_wheel_rpath_") as temp_dir:
    extracted_path = Path(temp_dir) / Path(members[0]).name
    extracted_path.write_bytes(payload)
    result = subprocess.run(
      ["patchelf", "--print-rpath", str(extracted_path)],
      capture_output=True,
      text=True,
      check=False)
  return result.stdout.strip()


def _normalize_wheel(wheel_path: Path, detected_platform_tag: str, retag: bool,
                     rpath: str | None) -> Path:
  head, python_tag, abi_tag, current_platform_tag = _parse_filename_platform_tag(wheel_path)
  new_platform_tag = detected_platform_tag if retag else current_platform_tag
  renaming = new_platform_tag != current_platform_tag
  if rpath is None and not renaming:
    return wheel_path

  with tempfile.TemporaryDirectory(prefix="cache_dit_wheel_normalize_") as temp_dir:
    extract_root = Path(temp_dir) / "wheel"
    extract_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(wheel_path) as wheel_zip:
      wheel_zip.extractall(extract_root)

    dist_info_dirs = list(extract_root.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
      raise RuntimeError(
        f"Expected exactly one dist-info directory in {wheel_path}, got {dist_info_dirs}")

    if rpath is not None:
      for elf_path in sorted(path for path in extract_root.rglob("*.so") if path.is_file()):
        subprocess.check_call(["patchelf", "--set-rpath", rpath, str(elf_path)])

    if renaming:
      wheel_metadata_path = dist_info_dirs[0] / "WHEEL"
      wheel_metadata_lines = wheel_metadata_path.read_text().splitlines()
      rewritten_lines = []
      for line in wheel_metadata_lines:
        if line.startswith("Tag: "):
          tag_python, tag_abi, _ = line[5:].rsplit("-", 2)
          rewritten_lines.append(f"Tag: {tag_python}-{tag_abi}-{new_platform_tag}")
          continue
        rewritten_lines.append(line)
      wheel_metadata_path.write_text("\n".join(rewritten_lines) + "\n")

    _rewrite_record(extract_root, dist_info_dirs[0])

    normalized_wheel_path = wheel_path.with_name(
      f"{head}-{python_tag}-{abi_tag}-{new_platform_tag}.whl")
    with ZipFile(normalized_wheel_path, "w", compression=ZIP_DEFLATED) as wheel_zip:
      for file_path in sorted(path for path in extract_root.rglob("*") if path.is_file()):
        wheel_zip.write(file_path, file_path.relative_to(extract_root).as_posix())

  if normalized_wheel_path != wheel_path:
    wheel_path.unlink()
  return normalized_wheel_path


def main() -> None:
  args = _parse_args()
  detected_platform_tag = _detect_wheel_platform_tag(args.wheel)
  output_path = _normalize_wheel(args.wheel, detected_platform_tag, args.retag, args.set_rpath)
  line = f"{output_path}\t{detected_platform_tag}"
  if args.print_rpath:
    line += f"\t{_detect_wheel_rpath(output_path)}"
  print(line)


if __name__ == "__main__":
  main()

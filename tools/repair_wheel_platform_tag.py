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
    "Inspect a wheel's ELF payload and optionally retag it to the required manylinux floor.")
  parser.add_argument("wheel", type=Path, help="Path to the wheel file to inspect.")
  parser.add_argument(
    "--retag",
    action="store_true",
    help="Rewrite the wheel filename and WHEEL metadata to match the detected platform tag.",
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


def _retag_wheel(wheel_path: Path, detected_platform_tag: str) -> Path:
  head, python_tag, abi_tag, current_platform_tag = _parse_filename_platform_tag(wheel_path)
  if current_platform_tag == detected_platform_tag:
    return wheel_path

  with tempfile.TemporaryDirectory(prefix="cache_dit_wheel_retag_") as temp_dir:
    extract_root = Path(temp_dir) / "wheel"
    extract_root.mkdir(parents=True, exist_ok=True)
    with ZipFile(wheel_path) as wheel_zip:
      wheel_zip.extractall(extract_root)

    dist_info_dirs = list(extract_root.glob("*.dist-info"))
    if len(dist_info_dirs) != 1:
      raise RuntimeError(
        f"Expected exactly one dist-info directory in {wheel_path}, got {dist_info_dirs}")

    wheel_metadata_path = dist_info_dirs[0] / "WHEEL"
    wheel_metadata_lines = wheel_metadata_path.read_text().splitlines()
    rewritten_lines = []
    for line in wheel_metadata_lines:
      if line.startswith("Tag: "):
        tag_python, tag_abi, _ = line[5:].rsplit("-", 2)
        rewritten_lines.append(f"Tag: {tag_python}-{tag_abi}-{detected_platform_tag}")
        continue
      rewritten_lines.append(line)
    wheel_metadata_path.write_text("\n".join(rewritten_lines) + "\n")

    _rewrite_record(extract_root, dist_info_dirs[0])

    retagged_wheel_path = wheel_path.with_name(
      f"{head}-{python_tag}-{abi_tag}-{detected_platform_tag}.whl")
    with ZipFile(retagged_wheel_path, "w", compression=ZIP_DEFLATED) as wheel_zip:
      for file_path in sorted(path for path in extract_root.rglob("*") if path.is_file()):
        wheel_zip.write(file_path, file_path.relative_to(extract_root).as_posix())

  wheel_path.unlink()
  return retagged_wheel_path


def main() -> None:
  args = _parse_args()
  detected_platform_tag = _detect_wheel_platform_tag(args.wheel)
  output_path = _retag_wheel(args.wheel, detected_platform_tag) if args.retag else args.wheel
  print(f"{output_path}\t{detected_platform_tag}")


if __name__ == "__main__":
  main()

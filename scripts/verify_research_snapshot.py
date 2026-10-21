"""Read-only SHA-256 verification of the 2026-10-21 research publication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_records(base: Path, records: list[dict]) -> tuple[int, list[str]]:
    failures = []
    checked = 0
    for record in records:
        target = (base / record["path"]).resolve()
        if not target.is_relative_to(base.resolve()):
            failures.append(f"Path escapes archive: {record['path']}")
            continue
        if not target.is_file():
            failures.append(f"Missing: {record['path']}")
            continue
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        checked += 1
        if digest != record["sha256"]:
            failures.append(f"SHA-256 mismatch: {record['path']}")
    return checked, failures


def main() -> int:
    snapshot = json.loads((ROOT / "docs/git_sync_20261021/SHA256SUMS.json").read_text(encoding="utf-8-sig"))
    count, failures = check_records(ROOT, snapshot["files"])
    archive = ROOT / "archives/research_sync_20261021"
    imported = json.loads((archive / "IMPORT_MANIFEST.json").read_text(encoding="utf-8-sig"))
    copied_count, copied_failures = check_records(archive, imported["files"])
    failures.extend(copied_failures)
    print(f"Publication files checked: {count}")
    print(f"Imported original files checked: {copied_count}")
    print(f"External payloads recorded (not checked here): {len(imported['external_payloads'])}")
    for failure in failures:
        print(failure)
    print("PASS" if not failures else "FAIL")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

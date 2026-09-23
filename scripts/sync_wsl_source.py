"""Stage and verify the execution copy; preserve every previous copy.

Invoke under the exclusive runtime .source.lock (the bootstrap does this).
This copies source, not production datasets. No history directory is deleted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

EXCLUDED = {'.git', '.venv', 'node_modules', '__pycache__', '.pytest_cache'}


def synchronize(source: Path, runtime: Path) -> dict:
    source, runtime = source.resolve(), runtime.resolve()
    if not (source / 'app' / 'dft_engine.py').is_file():
        raise ValueError(f'Not an OFDFT source repository: {source}')
    if runtime == Path(runtime.anchor) or len(runtime.parts) < 4:
        raise ValueError(f'Unsafe runtime: {runtime}')
    if source == runtime or runtime in source.parents or source in runtime.parents:
        raise ValueError('Source and runtime must not contain each other')
    runtime.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.repo-sync-', dir=runtime))
    records = []
    for path in sorted(source.rglob('*')):
        rel = path.relative_to(source)
        if any(part in EXCLUDED for part in rel.parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink requires explicit review: {path}; stage retained at {stage}')
        if not path.is_file():
            continue
        raw = path.read_bytes()
        copied = raw.replace(b'\r\n', b'\n') if path.suffix == '.sh' else raw
        destination = stage / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(copied)
        shutil.copymode(path, destination)
        digest = hashlib.sha256(copied).hexdigest()
        if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
            raise IOError(f'Copy verification failed: {destination}')
        records.append({'path': rel.as_posix(), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                        'runtime_sha256': digest, 'shell_lf_normalized': raw != copied})
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report = {'created_utc': stamp, 'source': str(source), 'files': records}
    (stage / 'SOURCE_SYNC_MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    target, backup = runtime / 'repo', runtime / 'repo-history' / stamp
    if target.exists():
        if target.is_symlink():
            raise ValueError('Execution repo must not be a symlink')
        backup.parent.mkdir(exist_ok=True)
        target.rename(backup)
    try:
        stage.rename(target)
    except Exception:
        if backup.exists() and not target.exists():
            backup.rename(target)
        raise
    return {'files_verified': len(records), 'repository': str(target),
            'previous_copy': str(backup) if backup.exists() else None}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--runtime', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(synchronize(args.source, args.runtime), indent=2))

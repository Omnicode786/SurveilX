"""Coalesce byte-identical immutable prepared images; preserve paths and hashes.

Original datasets, weights, labels and active campaign directories are excluded.
Hardlinks share immutable bytes; edit an independent copy, never write in place.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def deduplicate(root, apply=False, excluded_prefix='yolo-g10'):
    root = Path(root).resolve(strict=True)
    files, sizes = [], {}
    for run in root.iterdir():
        if not run.is_dir() or run.name.startswith(excluded_prefix):
            continue
        for path in (run / 'dataset/images').glob('*/*'):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg'} or not path.is_file():
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('Prepared image escapes the selected runs directory')
            stat = path.stat()
            files.append((path, stat.st_size, stat.st_nlink))
            sizes[stat.st_size] = sizes.get(stat.st_size, 0) + 1
    # Prefer existing shared cache inodes; do not strand a cache's physical copy.
    files.sort(key=lambda row: (-row[2], str(row[0])))
    seen, changes = {}, []
    for path, size, _ in files:
        if sizes[size] < 2:
            continue
        fingerprint = digest(path)
        key = (size, fingerprint)
        canonical = seen.setdefault(key, path)
        if canonical.stat().st_ino == path.stat().st_ino:
            continue
        if canonical.stat().st_dev != path.stat().st_dev:
            continue
        row = {'path': str(path.relative_to(root)), 'canonical': str(canonical.relative_to(root)),
               'sha256': fingerprint, 'logical_bytes': size}
        if apply:
            if digest(canonical) != fingerprint or digest(path) != fingerprint:
                raise ValueError('Immutable prepared image changed during deduplication')
            temporary = path.with_name(f'.prepared-link-{uuid.uuid4().hex}')
            assert path.resolve().is_relative_to(root) and temporary.parent.resolve().is_relative_to(root)
            try:
                os.link(canonical, temporary)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            if digest(path) != fingerprint:
                raise ValueError('Prepared-image content changed')
        changes.append(row)
    return {'scope': 'Immutable generated image copies only; all paths/content preserved',
        'applied': apply, 'root': str(root), 'excluded_prefix': excluded_prefix,
        'images_scanned': len(files), 'coalesced_paths': len(changes), 'changes': changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/runs'))
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    report = deduplicate(args.root, args.apply)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'changes'}, indent=2))


if __name__ == '__main__':
    main()

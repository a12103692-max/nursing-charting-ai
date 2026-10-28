"""Explicit, synthetic-only Gold snapshots. No API calls or training changes."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

from .core import load_jsonl, validate


FORMAT = 'nursing-charting-synthetic-gold-v1'
ROW_FIELDS = {'id', 'synthetic', 'input', 'context', 'records', 'source', 'review', 'pattern'}
REVIEW_FIELDS = {'reviewer', 'approved_at_utc', 'draft_sha256', 'notes'}


def checksum(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def check_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('비어 있지 않은 합성 학습 Gold 목록이 필요합니다.')
    for row in rows:
        validate(row)
        if set(row) - ROW_FIELDS:
            raise ValueError('Gold 외 필드가 있습니다. 초안/설정/키 파일은 백업할 수 없습니다.')
        if 'review' in row and (not isinstance(row['review'], dict) or set(row['review']) - REVIEW_FIELDS):
            raise ValueError('알 수 없는 승인 메타데이터 필드입니다.')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('중복 ID가 있습니다.')
    # A narrow accidental-secret check, not a patient-information detector.
    serialized = json.dumps(rows, ensure_ascii=False)
    if re.search(r'\bsk-[A-Za-z0-9_-]{16,}|\bBearer\s+\S+', serialized):
        raise ValueError('자격증명으로 의심되는 문자열이 있습니다. 백업을 중단합니다.')
    return rows


def write_new(path, text):
    """Publish a complete file atomically; never replace an existing target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            os.unlink(temporary)


def export_gold(store, output, *, confirmed):
    if confirmed is not True:
        raise ValueError('--confirm-synthetic으로 실제 환자정보가 없는 합성 학습 Gold임을 확인하세요.')
    rows = check_rows(load_jsonl(store))
    payload = {'format': FORMAT, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'notice': 'Synthetic training Gold only; no evaluation cases, credentials or draft sessions.',
               'count': len(rows), 'sha256': checksum(rows), 'rows': rows}
    write_new(output, json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    return {'count': len(rows), 'sha256': payload['sha256']}


def restore_gold(backup, store):
    payload = json.loads(Path(backup).read_text(encoding='utf-8'))
    if payload.get('format') != FORMAT:
        raise ValueError('지원하지 않는 백업 형식입니다.')
    rows = check_rows(payload['rows'])
    if type(payload.get('count')) is not int or payload['count'] != len(rows) or payload.get('sha256') != checksum(rows):
        raise ValueError('건수 또는 체크섬이 다릅니다. 복원하지 않습니다.')
    write_new(store, ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return {'count': len(rows), 'sha256': checksum(rows)}


def main():
    parser = argparse.ArgumentParser(description='합성 학습 Gold만 백업/빈 저장 경로에 복원')
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export')
    export.add_argument('--store', type=Path, default=Path('.nursing-data/gold.jsonl'))
    export.add_argument('--output', type=Path, required=True)
    export.add_argument('--confirm-synthetic', action='store_true')
    restore = commands.add_parser('restore')
    restore.add_argument('--backup', type=Path, required=True)
    restore.add_argument('--store', type=Path, default=Path('.nursing-data/gold.jsonl'))
    args = parser.parse_args()
    try:
        result = (export_gold(args.store, args.output, confirmed=args.confirm_synthetic)
                  if args.command == 'export' else restore_gold(args.backup, args.store))
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Do not print malformed input or credential-like strings from exceptions.
        print(f'백업/복원 실패 ({type(exc).__name__}). 입력·체크섬·저장 경로 중복을 확인하세요.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

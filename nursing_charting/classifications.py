"""Read-only human-reviewed Gold classification audit (no record generation).

The manifest labels source examples, not clinical facts or authorization to
expand new inputs. A mismatch in source identity or review metadata fails closed.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .backup import FORMAT as GOLD_FORMAT, check_rows, checksum


FORMAT = 'nursing-charting-human-classification-v1'
CATEGORIES = frozenset({
    'set_charting', 'general_transform', 'conditional_restore',
    'previous_record_reference', 'input_typo',
})
APPROVED_BASIS = frozenset({
    'user_reviewed', 'prior_user_review', 'set_rereview',
    'user_approved_2026-10-23',
})


def _read_object(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(obj, dict):
        raise ValueError('Classification/Gold document must be a JSON object.')
    return obj


class ClassificationIndex:
    """A verified mapping between immutable Gold rows and human decisions."""

    def __init__(self, *, rows, labels, source_hash):
        self.rows = rows
        self.labels = labels
        self.source_hash = source_hash
        self.by_input = defaultdict(list)
        for row in rows:
            self.by_input[(row['context'], row['input'])].append(row)

    def summary(self):
        return {'gold_count': len(self.rows), 'classification_counts':
                dict(sorted(Counter(x['classification'] for x in self.labels.values()).items())),
                'gold_rows_sha256': self.source_hash, 'validated': True}

    def bind(self, rows):
        """Keep snapshot decisions bound to unchanged source rows, not just IDs."""
        original = {row['id']: row for row in self.rows}
        for row in rows:
            if row['id'] in original and row != original[row['id']]:
                raise ValueError('Reviewed Gold changed; classification must be reviewed again.')

    def category(self, row):
        entry = self.labels.get(row['id'])
        return entry['classification'] if entry else None

    def lookup(self, text, context=''):
        """Exact known inputs only; no nearest-match, inference, or chart output."""
        if not isinstance(text, str) or not isinstance(context, str):
            raise ValueError('Text and context must be strings.')
        found = self.by_input.get((context, text), [])
        if not found:
            return {'status': 'unclassified', 'classification': None,
                    'evidence_ids': [], 'requires_review': True,
                    'reason': 'No exact reviewed input/context; do not infer a Set trigger.'}
        ids = sorted(r['id'] for r in found)
        kinds = {self.labels[i]['classification'] for i in ids}
        answers = {tuple(row['records']) for row in found}
        if len(kinds) != 1 or len(answers) != 1:
            return {'status': 'conflict', 'classification': None,
                    'evidence_ids': ids, 'requires_review': True,
                    'reason': 'Conflicting classifications or Gold answers: abstain.'}
        category = kinds.pop()
        if category in ('conditional_restore', 'previous_record_reference', 'input_typo'):
            status = 'human_check_required'
        elif category == 'set_charting':
            status = 'reviewed_set_example'
        else:
            status = 'reviewed_general_example'
        return {'status': status, 'classification': category,
                'evidence_ids': ids, 'requires_review': True,
                'reason': ('Classification is not evidence an intervention occurred. '
                           'No chart record was generated or approved.')}


def load_index(manifest_path, backup_path):
    manifest = _read_object(manifest_path)
    backup = _read_object(backup_path)
    if manifest.get('format') != FORMAT or backup.get('format') != GOLD_FORMAT:
        raise ValueError('Unsupported manifest or Gold snapshot format.')
    rows = check_rows(backup.get('rows'))
    if type(backup.get('count')) is not int or backup['count'] != len(rows):
        raise ValueError('Gold count mismatch.')
    digest = checksum(rows)
    if digest != backup.get('sha256'):
        raise ValueError('Gold checksum mismatch.')
    source = manifest.get('gold_source')
    if not isinstance(source, dict) or source.get('format') != GOLD_FORMAT or \
            type(source.get('count')) is not int or source['count'] != len(rows) or \
            source.get('rows_sha256') != digest:
        raise ValueError('Manifest is not bound to the supplied Gold snapshot.')
    entries = manifest.get('rows')
    if not isinstance(entries, list) or len(entries) != len(rows):
        raise ValueError('Manifest entry count mismatch.')
    gold_ids = {r['id'] for r in rows}
    labels = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('Invalid classification entry.')
        identifier = entry.get('id')
        if not isinstance(identifier, str) or identifier not in gold_ids or identifier in labels:
            raise ValueError('Duplicate/unknown classification ID.')
        if entry.get('classification') not in CATEGORIES:
            raise ValueError('Unrecognized classification.')
        if entry.get('review_basis') not in APPROVED_BASIS:
            raise ValueError('Classification has no accepted human review basis.')
        labels[identifier] = entry
    if set(labels) != gold_ids:
        raise ValueError('Manifest is missing Gold rows.')
    declared = manifest.get('counts')
    totals = Counter(r['classification'] for r in labels.values())
    if not isinstance(declared, dict) or set(declared) != {'total', *CATEGORIES} or \
            type(declared['total']) is not int or declared['total'] != len(rows) or \
            any(type(declared[k]) is not int or declared[k] != totals[k] for k in CATEGORIES):
        raise ValueError('Declared classification counts do not match the rows.')
    meta = manifest.get('review_provenance', {})
    if not isinstance(meta, dict) or meta.get('assistant_inferred_general_unverified') != 0:
        raise ValueError('Unverified assistant guesses must not be treated as approved.')
    # The 2026-10-23 Set-62 decision is explicitly traced to this row.
    review = meta.get('set_rereview')
    if not isinstance(review, dict) or review.get('conditional_restore_id') != 'C06-R07' or \
            labels.get('C06-R07', {}).get('classification') != 'conditional_restore':
        raise ValueError('Missing or inconsistent C06-R07 human-review provenance.')
    approval = meta.get('additional_general_approval_2026_10_23')
    if not isinstance(approval, dict) or approval.get('total') != 25:
        raise ValueError('Missing 25-case approval history.')
    ids10, ids15 = approval.get('unit_format_batch_10'), approval.get('remaining_batch_15')
    if not isinstance(ids10, list) or not isinstance(ids15, list) or len(ids10) != 10 or len(ids15) != 15:
        raise ValueError('Invalid 25-case approval partition.')
    approved = ids10 + ids15
    if len(set(approved)) != 25 or any(labels.get(i, {}).get('classification') != 'general_transform' or
                                         labels[i].get('review_basis') != 'user_approved_2026-10-23'
                                         for i in approved):
        raise ValueError('25-case approvals disagree with manifest rows.')
    if {i for i, x in labels.items() if x.get('review_basis') == 'user_approved_2026-10-23'} != set(approved):
        raise ValueError('Unexpected approved-batch membership.')
    return ClassificationIndex(rows=rows, labels=labels, source_hash=digest)


def runtime_index(rows):
    """Use the checked-in decisions for stores containing snapshot examples.

    Additional local Gold stays intact and usable as exact/style evidence, but
    receives no Set classification or automatic rule support by implication.
    Independent synthetic stores retain their existing behavior.
    """
    root = Path(__file__).resolve().parent.parent
    manifest = root / 'reports/set-charting-human-classifications-2026-10-23.json'
    backup = root / 'backups/synthetic-training-gold-2026-10-22.json'
    if not (manifest.is_file() and backup.is_file()):
        return None  # Public demo has no private classification manifest / Gold snapshot.
    index = load_index(manifest, backup)
    index.bind(rows)
    return index if any(row['id'] in index.labels for row in rows) else None


def main(argv=None):
    parser = argparse.ArgumentParser(description='Read-only review classification audit; no chart generation')
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--input', help='Optional exact synthetic input to look up')
    parser.add_argument('--context', default='')
    args = parser.parse_args(argv)
    try:
        index = load_index(args.manifest, args.backup)
        result = index.summary() if args.input is None else index.lookup(args.input, args.context)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f'분류 검증 실패: {type(exc).__name__}. Gold와 승인된 manifest를 확인하세요.\n')


if __name__ == '__main__':
    raise SystemExit(main())

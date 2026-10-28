"""Small draft/review boundary around the existing model and API adapter."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

from .core import Model, key, validate
from .classifications import runtime_index
from .quality import select_style_gold, normalize_draft


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def draft_view(session, gold_count):
    """Shared presentation for single drafts and persisted queue entries."""
    snapshot = session['snapshot']
    prediction = snapshot['prediction']
    decision = snapshot.get('gold_classifications', {}).get('decision', {})
    uncertainties = list(prediction['uncertainties'])
    diagnostics = deepcopy(snapshot.get('gold_diagnostics', {}))
    # Read-only presentation for older queues; never rewrite signed snapshots.
    if 'gold_diagnostics' not in snapshot:
        reviews = snapshot.get('gold_conflict_reviews', {})
        path = Path(__file__).resolve().parent.parent / 'gold_conflict_reviews.json'
        notes = [r.get('review_note', '') for r in json.loads(path.read_text(encoding='utf-8'))['reviews']]
        management = [n for n in notes if n and n in uncertainties]
        management += [i['reason'] + ' 그룹: ' + ', '.join(i['group_ids'])
                       for i in reviews.get('invalidated', [])]
        uncertainties = [q for q in uncertainties if q not in management]
        diagnostics = dict(reviews, review_notes=management,
                           excluded_conflicting_ids=snapshot['excluded_conflicting_ids'])
    return {'id': snapshot['origin']['id'], 'route': prediction['route'],
            'classification': decision.get('classification'),
            'classification_status': decision.get('status', 'unclassified'),
            'llm_attempted': prediction['llm_attempted'],
            'records': prediction['records'], 'uncertainties': uncertainties,
            'omissions': prediction['omissions'], 'reason': prediction.get('reason', ''),
            'workflow_notices': prediction.get('workflow_notices', []),
            'diagnostics': diagnostics,
            'evidence_ids': prediction.get('evidence_ids', []),
            'gold_ids_sent': snapshot['gold_ids_sent'],
            'gold_count': gold_count, 'gold_sent': len(snapshot['gold_ids_sent']),
            'excluded_conflicting_ids': snapshot['excluded_conflicting_ids'],
            'usage': snapshot['api_calls'][-1].get('usage') if snapshot['api_calls'] else None}


def conflicting_ids(rows):
    groups = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    return sorted(r['id'] for group in groups.values()
                  if len({tuple(r['records']) for r in group}) > 1 for r in group)


def group_fingerprint(rows):
    return fingerprint(sorted(rows, key=lambda row: row['id']))


def select_llm_gold(rows, reviews_path=None):
    """Apply only explicit human representative selections; never merge facts."""
    excluded = set(conflicting_ids(rows))
    path = (Path(reviews_path) if reviews_path is not None else
            Path(__file__).resolve().parent.parent / 'gold_conflict_reviews.json')
    applied, invalidated, notes = [], [], []
    if not path.exists():
        return [r for r in rows if r['id'] not in excluded], sorted(excluded), applied, invalidated, notes
    document = json.loads(path.read_text(encoding='utf-8'))
    if document.get('version') != 1 or not isinstance(document.get('reviews'), list):
        raise ValueError('지원하지 않는 Gold 충돌 검토 파일입니다.')
    indexed = {r['id']: r for r in rows}
    claimed = set()
    for review in document['reviews']:
        if not isinstance(review, dict):
            raise ValueError('충돌 검토 항목은 JSON 객체여야 합니다.')
        ids = review.get('group_ids')
        if (not isinstance(ids, list) or len(ids) < 2 or any(not isinstance(i, str) for i in ids)
                or len(set(ids)) != len(ids) or claimed.intersection(ids)):
            raise ValueError('충돌 검토 그룹 ID가 잘못되었거나 중복됩니다.')
        claimed.update(ids)
        if not set(ids).intersection(indexed):
            continue
        members = [indexed[i] for i in ids if i in indexed]
        current_group = [r for r in rows if members and key(r) == key(members[0])]
        representative = review.get('representative_id')
        valid = (review.get('approved') is True and representative in ids
                 and set(ids) == {r['id'] for r in current_group}
                 and set(ids).issubset(excluded)
                 and group_fingerprint(current_group) == review.get('group_sha256'))
        if not valid:
            invalidated.append({'group_ids': ids, 'reason': '원본 그룹/해시 또는 승인 정보가 달라 대표 선택을 적용하지 않았습니다.'})
            continue
        excluded.remove(representative)
        applied.append({'group_ids': ids, 'representative_id': representative,
                        'group_sha256': review['group_sha256']})
        note = review.get('review_note', '')
        if not isinstance(note, str):
            raise ValueError('검토 안내는 문자열이어야 합니다.')
        if note:
            notes.append(note)
    return [r for r in rows if r['id'] not in excluded], sorted(excluded), applied, invalidated, notes


def draft(rows, text, context, client, *, synthetic, reviews_path=None):
    if synthetic is not True:
        raise ValueError('합성 입력만 허용합니다. --synthetic을 명시하세요.')
    if not isinstance(text, str) or not text.strip() or not isinstance(context, str):
        raise ValueError('비어 있지 않은 입력과 문자열 맥락이 필요합니다.')
    index = runtime_index(rows)
    model = Model(rows, classification_index=index)
    prediction = model.transform(text, context)
    gold, excluded, applied, invalidated, review_notes = select_llm_gold(rows, reviews_path)
    classification_excluded = []
    if index is not None:
        classification_excluded = [r['id'] for r in gold if index.category(r) in
                                   ('conditional_restore', 'previous_record_reference', 'input_typo')]
        gold = [r for r in gold if r['id'] not in classification_excluded]
    eligible_count = len(gold)
    gold = select_style_gold(gold, text, context, index)
    used = []
    unit_changes = []
    workflow_notices = []
    if prediction.get('classification_status') == 'human_check_required':
        questions = {
            'conditional_restore': '조건부 Gold입니다. 생략된 상태·중재의 근거와 시행/계획 여부를 확인해 직접 작성하세요.',
            'previous_record_reference': '이전 기록 참조 사례입니다. 현재 적용할 이전 기록과 상태를 확인해 직접 작성하세요.',
            'input_typo': '입력 오타 검토 사례입니다. 의도한 입력을 확인해 다시 입력하세요.',
        }
        prediction = dict(prediction, llm_attempted=False,
                          uncertainties=[questions[prediction['classification']]], omissions=[])
    elif prediction['route'] == 'needs_review' and prediction['evidence_ids']:
        # A known conflict must never be resolved by asking an LLM to pick a side.
        prediction = dict(prediction, route='conflict', llm_attempted=False,
                          uncertainties=[prediction['reason']], omissions=[])
    elif prediction['route'] == 'needs_review':
        used = [r['id'] for r in gold]
        options = {'classification_index': index} if index is not None else {}
        generated = deepcopy(client.generate(text, context, gold, **options))
        generated, unit_changes = normalize_draft(generated, text, context)
        if client.calls:
            unit_changes = client.calls[-1].get('notation_changes', unit_changes)
        if index is not None:
            workflow_notices.append(
                '현재 입력은 승인된 Set 원문·맥락과 정확히 일치하지 않습니다. Gold의 생략된 상태·중재를 '
                '새 Set으로 자동 복원하지 마세요. 임상 사실과 시행 여부를 검토하세요.')
        prediction = dict(generated, route='llm' if generated['records'] else 'needs_review',
                          requires_review=True, llm_attempted=True,
                          gold_example_count=len(gold), revision=model.revision)
    else:
        checks = ([] if prediction['route'] == 'exact' else [
            'Learned는 Gold의 고정 문구·단위도 재사용합니다. 입력/맥락에 없는 단위·상태·처치가 '
            '현재 사례에도 성립하는지 근거 Gold와 함께 확인한 뒤 승인하세요.'
        ])
        prediction = dict(prediction, llm_attempted=False, uncertainties=checks, omissions=[])
        if prediction.get('classification') == 'set_charting':
            prediction['uncertainties'].append(
                '승인된 Set 예시의 전체 문구를 재현한 미승인 초안입니다. 원문에서 생략된 단위·경로·상태·처치·교육 등이 '
                '현재 사례에도 실제 관찰·시행되었는지 확인하고 수정한 뒤 승인하세요.')
    prediction['workflow_notices'] = workflow_notices
    origin = {'id': 'review-' + uuid4().hex, 'synthetic': True,
              'input': text, 'context': context, 'source': 'reviewed'}
    snapshot = {'origin': origin, 'prediction': deepcopy(prediction),
                'training_revision': model.revision, 'gold_ids_sent': used,
                'excluded_conflicting_ids': excluded, 'api_calls': deepcopy(client.calls)}
    snapshot['gold_diagnostics'] = {'review_notes': review_notes, 'invalidated': invalidated,
                                    'excluded_conflicting_ids': excluded,
                                    'eligible_style_count': eligible_count, 'selected_count': len(used),
                                    'selection_policy': 'source-context-relevance-v1; max 12; no unrelated padding',
                                    'unit_changes': unit_changes}
    snapshot['gold_conflict_reviews'] = {'applied': applied, 'invalidated': invalidated}
    if index is not None:
        snapshot['gold_classifications'] = {
            'gold_rows_sha256': index.source_hash,
            'decision': index.lookup(text, context),
            'excluded_from_llm': classification_excluded,
            'sent': {r['id']: index.category(r) or 'unclassified_local' for r in gold} if used else {},
        }
    return {'status': 'pending_review', 'snapshot': snapshot,
            'snapshot_sha256': fingerprint(snapshot),
            'review': {'records': list(prediction['records']), 'notes': ''}}


def approve(rows, session, *, approved, reviewer):
    if approved is not True or not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('--approve 및 --reviewer로 수정 결과를 명시적으로 승인하세요.')
    if session.get('status') not in ('pending_review', 'conflict'):
        raise ValueError('아직 승인하지 않은 초안만 저장할 수 있습니다.')
    snapshot = session['snapshot']
    if fingerprint(snapshot) != session['snapshot_sha256']:
        raise ValueError('원본 입력/초안이 변경되었습니다. review.records와 review.notes만 수정하세요.')
    if snapshot['prediction']['route'] == 'conflict':
        raise ValueError('기존 Gold 충돌을 먼저 검토·해결한 뒤 초안을 다시 생성하세요.')
    review = session['review']
    if not isinstance(review.get('notes', ''), str):
        raise ValueError('검토 메모는 문자열이어야 합니다.')
    candidate = validate(dict(snapshot['origin'], records=review['records']))
    if len(candidate['records']) != 1:
        raise ValueError('이번 수정·승인 흐름은 단일 기록만 지원합니다.')
    if any(r['id'] == candidate['id'] for r in rows):
        raise ValueError('이미 존재하는 ID입니다. 기존 Gold를 덮어쓰지 않습니다.')
    same = [r['id'] for r in rows if key(r) == key(candidate) and r['records'] != candidate['records']]
    prior = Model(rows).transform(candidate['input'], candidate['context'])
    if prior['route'] == 'learned' and prior['records'] != candidate['records']:
        same.extend(prior['evidence_ids'])
    if prior['route'] == 'needs_review' and prior['evidence_ids']:
        same.extend(prior['evidence_ids'])
    if same:
        session['status'] = 'conflict'
        session['conflict'] = {'existing_ids': sorted(set(same)), 'candidate': candidate,
                               'reason': '기존 Exact/Learned 정답과 상충함. Gold 저장 안 함.'}
        return None
    candidate['review'] = {'reviewer': reviewer.strip(),
                           'approved_at_utc': datetime.now(timezone.utc).isoformat(),
                           'draft_sha256': session['snapshot_sha256'],
                           'notes': review.get('notes', '')}
    session['status'] = 'approved'
    session.pop('conflict', None)
    session['approved_gold'] = candidate
    return candidate

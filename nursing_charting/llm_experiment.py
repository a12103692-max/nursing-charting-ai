"""One paired experiment: Exact, then LLM without/with training Gold."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from .core import Model, key, validate
from .llm import LLMError, exact_result, transform


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def numbers(texts):
    # Screening only: this does not know units, clinical fields, or equivalence.
    return Counter(re.findall(r'(?<![A-Za-z0-9_])\d+(?:\.\d+)?(?![A-Za-z0-9_])', '\n'.join(texts)))


def compare(expected, prediction):
    if prediction.get('records') is None:
        return {'string_exact_match': None, 'numeric_token_screen': None,
                'clinical_fact_agreement': 'not_assessed_no_output'}
    actual = prediction['records']
    gold_numbers, output_numbers = numbers(expected), numbers(actual)
    return {'string_exact_match': actual == expected,
            'numeric_token_screen': {
                'exact_multiset_match': gold_numbers == output_numbers,
                'missing_from_output': dict(gold_numbers - output_numbers),
                'additional_in_output': dict(output_numbers - gold_numbers)},
            'clinical_fact_agreement': 'requires_human_review' if actual else 'not_assessed_abstained'}


def run(training, cases, client, output, max_calls=50):
    output = Path(output)
    if output.exists():
        raise ValueError('기존 보고서와 다른 출력 경로를 지정하세요.')
    if not training or not cases:
        raise ValueError('비어 있지 않은 학습 Gold와 평가 사례가 필요합니다.')
    if max_calls < 0:
        raise ValueError('max-calls must be nonnegative')
    for row in training + cases:
        validate(row)
    training_keys = {key(row) for row in training}
    model = Model(training, automatic=False)
    replay = [{'id': row['id'], 'expected': row['records'],
               'prediction': exact_result(training, row['input'], row['context']) or {
                   'route': 'needs_review', 'records': [], 'reason': '학습 정답 충돌'},
               } for row in training]
    report = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'running', 'requested_model': client.model,
        'training_revision': model.revision, 'training_sha256': fingerprint(training),
        'evaluation_sha256': fingerprint(cases),
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
        'gold_strategy': 'all_training_pairs_in_context; no fine-tuning; no evaluation expected records in requests',
        'automatic_pattern_engine_used': False,
        'max_generation_calls': max_calls,
        'planned_generation_calls': 2 * sum(exact_result(training, r['input'], r['context']) is None for r in cases),
        'training_exact_replay': {'count': len(replay),
            'matched': sum(r['prediction']['records'] == r['expected'] for r in replay), 'results': replay},
        'evaluation': [], 'api_calls': client.calls,
        'limitations': [
            'Evidence validation checks quote presence, not clinical entailment or absence of hallucinations.',
            'Numeric token matching ignores field roles, units and clinical meaning; decimal commas are not repaired.',
            'LLM uncertainties/omissions are self reports, not a complete clinical audit.',
            'Already inspected 27-case evaluation is not a new blind benchmark; no expected records enter requests.',
            'Row-level data does not validate cross-row Set Charting.',
            'One sample per condition; temperature zero does not guarantee deterministic output.',
        ],
    }
    stopped = None if client.configured else 'missing_NURSING_LLM_API_KEY_or_model'

    def persist():
        # A separate report; never use the Gold-writing function or write to its path.
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        temporary.replace(output)

    persist()
    for row in cases:
        # Expected records remain only in the local scorer/review report.
        item = {'id': row['id'], 'context': row['context'], 'input': row['input'],
                'expected': row['records'], 'seen_input_context': key(row) in training_keys}
        for condition, with_gold in [('without_gold', False), ('with_gold', True)]:
            replayed = exact_result(training, row['input'], row['context'])
            if replayed:
                prediction = dict(replayed, llm_attempted=False, uncertainties=[], omissions=[])
            elif stopped or len(client.calls) >= max_calls:
                prediction = {'route': 'not_run', 'records': None, 'llm_attempted': False,
                              'reason': stopped or 'generation_call_budget_exhausted'}
            else:
                try:
                    prediction = transform(training, row['input'], row['context'], client,
                                           synthetic=True, with_gold=with_gold)
                except LLMError as exc:
                    stopped = str(exc)
                    prediction = {'route': 'error', 'records': None, 'llm_attempted': True, 'reason': stopped}
            item[condition] = {'prediction': prediction, **compare(row['records'], prediction)}
            # Do not put expected records into transform() or client.generate().
        report['evaluation'].append(item)
        persist()
    summary = {}
    for condition in ('without_gold', 'with_gold'):
        values = [r[condition] for r in report['evaluation']]
        novel = [r[condition] for r in report['evaluation'] if not r['seen_input_context']]
        summary[condition] = {
            'count': len(values), 'string_exact_matches': sum(r['string_exact_match'] is True for r in values),
            'scored': sum(r['string_exact_match'] is not None for r in values),
            'routes': dict(Counter(r['prediction']['route'] for r in values)),
            'unseen_inputs': len(novel),
            'unseen_string_exact_matches': sum(r['string_exact_match'] is True for r in novel),
            'unseen_scored': sum(r['string_exact_match'] is not None for r in novel),
        }
    paired = [r for r in report['evaluation'] if all(
        r[c]['prediction'].get('llm_attempted') and r[c]['prediction']['route'] in ('llm', 'needs_review')
        for c in ('without_gold', 'with_gold'))]
    report['gold_effect'] = {
        'paired_llm_cases_completed': len(paired),
        'string_exact_match_count_difference': (
            sum(r['with_gold']['string_exact_match'] for r in paired) -
            sum(r['without_gold']['string_exact_match'] for r in paired)) if paired else None,
        'clinical_fact_agreement_difference': None,
        'clinical_review_note': 'Clinical equivalence and omissions require human review of actual records.',
    }
    usage = [r['usage'] for r in client.calls if isinstance(r.get('usage'), dict)]
    report['summary'] = summary
    report['api_measurement'] = {
        'generation_requests_attempted': len(client.calls),
        'http_200_responses': sum(r.get('http_status') == 200 for r in client.calls),
        'validated_responses': sum(r.get('status') == 'validated_structure_and_quote_presence_only' for r in client.calls),
        'usage_reported_responses': len(usage),
        'reported_prompt_tokens': sum(r.get('prompt_tokens', 0) for r in usage),
        'reported_completion_tokens': sum(r.get('completion_tokens', 0) for r in usage),
        'cost_usd': None, 'cost_note': 'No billing measurement or verified pricing supplied; see raw usage per call.',
    }
    report['status'] = 'incomplete' if any(
        x[condition]['prediction']['route'] in ('not_run', 'error')
        for x in report['evaluation'] for condition in ('without_gold', 'with_gold')) else 'completed'
    report['stop_reason'] = stopped
    persist()
    return report

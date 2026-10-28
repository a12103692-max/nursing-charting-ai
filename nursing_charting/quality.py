"""Deterministic style-reference selection and traceable notation policy.

No clinical restoration, classification, or Gold writes occur here.
"""
from collections import Counter
from copy import deepcopy
import re

MAX_STYLE_EXAMPLES = 12
# These are notation conventions evidenced by verified general_transform Gold.
# This does not authorize the other facts or medication units in those records.
UNIT_POLICY = {
    'HR': {'unit': '회/min', 'basis': ['C06-R02'], 'aliases': ['HR', '심박수', '심박동수']},
    'SBP': {'unit': 'mmHg', 'basis': ['C02-R06', 'C01-R09'], 'aliases': ['SBP']},
    'MBP': {'unit': 'mmHg', 'basis': ['C01-R09'], 'aliases': ['MBP']},
    'BP': {'unit': 'mmHg', 'basis': ['C17-R09'], 'aliases': ['BP']},
    'SpO₂': {'unit': '%', 'basis': ['C03-R07'], 'aliases': ['SpO2', 'SpO₂']},
}
PENDING_UNITS = {'RR': '호흡수 표기 규칙 승인 필요', 'MAP': 'MAP 별칭/단위 정책 승인 필요',
                 'PAP': '폐동맥압 단위 정책 승인 필요', 'PA': 'PA 수치의 의미/단위 확인 필요',
                 'CVP': 'CVP 단위 정책 승인 필요', 'K': '검사실 단위 확인 필요'}
NUMBER = r'\d+(?:\.\d+)?'
VALUE = rf'{NUMBER}(?:\s*[/~–-]\s*{NUMBER})?'
# Device concepts also match Korean descriptions. They rank style only.
CONCEPTS = [r'a[- ]?line|동맥관', r'iabp|대동맥.*풍선',
            r'pa\s*catheter|swan|폐동맥|\bcvp\b|\bpap\b',
            r'(?<![/\w])hr(?=\s*[\d(])|심박|맥박', r'spo[2₂]|산소포화',
            r'\b(?:sbp|mbp|map|bp)\b|혈압', r'\bk\b|전해질|칼륨',
            r'통증|\bnrs\b|pain', r'배액|\bjp\b|drain',
            r'지남력|의식|alert', r'프리세덱스|precedex', r'crrt|투석']


def terms(text):
    return {s for s in re.findall(r'[a-z][a-z0-9]+|[가-힣]{2,}', text.casefold())
            if s not in {'확인', '없음', '유지', '관찰', '시행', '예정', '있음',
                          'cm', 'mm', 'mg', 'ml', 'mcg', 'kg', 'min', 'hr', 'mmhg', 'cc', 'iv'}}


def select_style_gold(rows, text, context='', index=None, limit=MAX_STYLE_EXAMPLES):
    """After full-corpus safety checks: rank source/context, never target answers.

    No unrelated padding when fewer than limit examples are relevant. Zero is OK.
    Classification is an eligibility filter and a tie-break, never a new Set rule.
    """
    if not 1 <= limit <= MAX_STYLE_EXAMPLES:
        raise ValueError('Style example limit must be between 1 and 12.')
    if index is not None:
        index.bind(rows)
    current = text + ' ' + context
    query = terms(current)
    concepts = {i for i, p in enumerate(CONCEPTS) if re.search(p, current, re.I)}
    ranked = []
    for row in rows:
        category = index.category(row) if index else None
        if category in {'conditional_restore', 'previous_record_reference', 'input_typo'}:
            continue
        source = row['input'] + ' ' + row['context']
        shared = query & terms(source)
        related = {i for i in concepts if re.search(CONCEPTS[i], source, re.I)}
        if (concepts and not related) or not (shared or related):
            continue
        score = 4 * len(related) + len(shared) / max(1, len(query))
        ranked.append((score, category == 'general_transform', row['id'], row))
    ranked.sort(key=lambda x: (-x[0], -x[1], x[2]))
    return [x[3] for x in ranked[:limit]]


def measurement_pattern(aliases):
    label = '|'.join(re.escape(a) for a in aliases)
    return re.compile(rf'(?<![\w])(?:{label})\s*[:=]?\s*\(?\s*(?P<value>{VALUE})\s*\)?', re.I)


def normalize_draft(draft, text, context=''):
    """Normalize only explicitly named, supported values; retain raw draft elsewhere.

    Arbitrary dosages, lab values, bare numbers and inferred aliases are excluded.
    Existing incompatible units remain unchanged and are flagged for review.
    """
    result = deepcopy(draft)
    checks, changes = [], []
    source = text + '\n' + context
    for label, rule in UNIT_POLICY.items():
        pattern = measurement_pattern(rule['aliases'])
        unit_re = r'(?:회\s*/\s*(?:min|분)|bpm|mmHg|%|회)(?![a-zA-Z])'
        full = re.compile(pattern.pattern + rf'(?P<unit>\s*{unit_re})?'
                          + rf'(?P<chain>\s*에서\s*(?P<next>{VALUE})(?P<nextunit>\s*{unit_re})?)?', re.I)
        supported = set()
        allowed = {'회/min', '회/분', 'bpm', '회'} if label == 'HR' else {rule['unit'].casefold()}
        def compact(value):
            return re.sub(r'\s+', '', value).casefold()
        def ambiguous_slash(match):
            # Only BP has an approved systolic/diastolic pair interpretation.
            # Do not treat HR 80/100 as a range, rate or two ordered readings.
            values = [match['value'], match['next']]
            trailing_slash = re.match(r'\s*/', match.string[match.end():])
            if trailing_slash or (label != 'BP' and any(v and '/' in v for v in values)):
                checks.append(f'{label}: 슬래시 수치의 의미를 확인하세요. 자동 단위 표준화를 적용하지 않았습니다.')
                return True
            return False
        for match in full.finditer(source):
            if ambiguous_slash(match):
                continue
            tail = source[match.end():]
            units = [match['unit'], match['nextunit']]
            if any(u and compact(u) not in allowed for u in units) or re.match(
                    r'(?:\s*[a-zA-Z]+(?:/\w+)*|,\d)', tail):
                checks.append(f'{label}: 원문 단위 확인 필요. 자동 단위 변환을 적용하지 않았습니다.')
                continue
            supported.add(compact(match['value']))
            if match['next']:
                supported.add(compact(match['next']))
        if not supported:
            continue
        for i, record in enumerate(result['records']):
            def replace(match):
                if ambiguous_slash(match):
                    return match[0]
                values = [match['value']] + ([match['next']] if match['next'] else [])
                if any(compact(v) not in supported for v in values):
                    checks.append(f'{label}: 원문과 다른 측정값 확인 필요')
                    return match[0]
                if any(u and compact(u) not in allowed for u in [match['unit'], match['nextunit']]):
                    checks.append(f'{label}: 상충하는 단위 확인 필요')
                    return match[0]
                if re.match(r'(?:\s*(?:[a-zA-Z]+(?:/\w+)*|대)|,\d)', record[match.end():]):
                    checks.append(f'{label}: 단위 또는 근삿값 표기 확인 필요')
                    return match[0]
                result_text = f'{label} {match["value"].strip()} {rule["unit"]}'
                if match['next']:
                    result_text += f'에서 {match["next"].strip()} {rule["unit"]}'
                return result_text
            updated = full.sub(replace, record)
            if updated != record:
                changes.append({'record_index': i, 'measurement': label, 'basis': rule['basis']})
            result['records'][i] = updated
    explicit_unit = re.compile(r'\s*(?:mmHg|cmH[2₂]O|mmol/L|mEq/L|회\s*/\s*(?:min|분)|bpm|%)(?![a-zA-Z])', re.I)
    for label, reason in PENDING_UNITS.items():
        pattern = measurement_pattern([label])
        missing = []
        confirmed_units = []
        for match in pattern.finditer(source):
            stated_unit = explicit_unit.match(source[match.end():])
            if not stated_unit:
                missing.append(re.sub(r'\s+', '', match['value']))
            else:
                confirmed_units.append((match['value'], stated_unit[0].strip()))
        output = ' '.join(result['records'])
        for measured_value, unit in confirmed_units:
            if unit.casefold() not in output.casefold():
                checks.append(f'{label} {measured_value}: 원문 단위 {unit}의 출력 누락 여부를 확인하세요.')
        if not missing:
            continue
        checks.append(f'{label}: {reason}. 원문에 명시된 단위만 보존하세요.')
        # Strip only a precisely identified newly asserted unit, never a number.
        # This is not a conversion of lab values or an approval of unit policy.
        for i, record in enumerate(result['records']):
            unconfirmed = re.compile(pattern.pattern + r'(?P<unit>' + explicit_unit.pattern + ')', re.I)
            def remove_unconfirmed(match):
                if re.sub(r'\s+', '', match['value']) not in missing:
                    return match[0]
                checks.append(f'{label}: 원문에 없는 단위를 초안에서 제외했습니다. 확인 후 직접 작성하세요.')
                return match[0][:-len(match['unit'])].rstrip()
            updated = unconfirmed.sub(remove_unconfirmed, record)
            if updated != record:
                changes.append({'record_index': i, 'measurement': label, 'basis': [],
                                'action': 'removed_unconfirmed_unit'})
            result['records'][i] = updated
    def numbers(value):
        return Counter(re.findall(NUMBER, re.sub(r'spo[2₂]', 'SpO', value, flags=re.I)))
    source_numbers = numbers(source)
    output_numbers = numbers(' '.join(result['records']))
    # Screening only: numbers in labels (SpO2), dates, repeats can need human review.
    if output_numbers - source_numbers:
        checks.append('원문보다 추가되거나 반복된 수치가 있습니다. 측정값·단위·시간을 대조하세요.')
    if source_numbers - output_numbers:
        checks.append('원문의 일부 수치가 초안에 없습니다. 중요한 측정값·시간 누락 여부를 확인하세요.')
    result['uncertainties'] = list(dict.fromkeys(result['uncertainties'] + checks))
    return result, changes


def unit_guidance():
    return ('Deterministic approved notation (only explicitly named measurements, no new values): ' +
            '; '.join(f'{label}: {r["unit"]} (Gold {", ".join(r["basis"])})' for label, r in UNIT_POLICY.items()) +
            '. All other missing units, including RR/MAP/PAP/CVP/labs/drugs, require confirmation. '
            'Slash-valued measurements such as HR 80/100 are ambiguous: preserve the notation '
            'and ask for clarification; do not infer a range or add a rate unit. '
            'Only explicit BP systolic/diastolic pairs use the approved slash notation. '
            'Do not infer medication dose, concentration, route or infusion rate. '
            'Preserve vague ranges such as 90대; do not turn them into exact values.')

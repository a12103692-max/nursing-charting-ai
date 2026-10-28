"""Exact replay and explicitly approved template learning, using only supplied data."""

import hashlib
import json
import re
from collections import defaultdict
from string import Formatter

from .automatic import discover


def fields(template):
    names = []
    for _, name, spec, conversion in Formatter().parse(template):
        if name is not None:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or spec or conversion:
                raise ValueError("템플릿에는 {slot_name} 형식만 사용할 수 있습니다.")
            names.append(name)
    return names


def validate(row):
    if not isinstance(row, dict):
        raise ValueError("각 Gold는 JSON 객체여야 합니다.")
    for key in ("id", "input", "context"):
        if not isinstance(row.get(key), str) or (key != "context" and not row[key].strip()):
            raise ValueError(f"{key}: 문자열이 필요합니다(context는 빈 문자열 허용).")
    records = row.get("records")
    if not isinstance(records, list) or not records or any(not isinstance(x, str) or not x.strip() for x in records):
        raise ValueError("records: 비어 있지 않은 기록 문자열 배열이 필요합니다.")
    if row.get("synthetic") is not True:
        raise ValueError("합성 데이터만 허용합니다. synthetic: true를 명시하세요.")
    if row.get("source", "gold") not in ("gold", "reviewed"):
        raise ValueError("source는 gold 또는 reviewed여야 합니다.")
    pattern = row.get("pattern")
    if pattern is not None:
        if not isinstance(pattern, dict) or pattern.get("approved") is not True:
            raise ValueError("일반화 템플릿은 pattern.approved: true로 사람이 확인해야 합니다.")
        template = pattern.get("input_template")
        outputs = pattern.get("record_templates")
        bindings = pattern.get("bindings")
        if not isinstance(template, str) or not isinstance(outputs, list) or not outputs or any(not isinstance(x, str) for x in outputs):
            raise ValueError("input_template 및 record_templates가 필요합니다.")
        names = fields(template)
        if not names or len(names) != len(set(names)):
            raise ValueError("입력 슬롯은 하나 이상이고 각각 한 번만 나타나야 합니다.")
        if not isinstance(bindings, dict) or set(bindings) != set(names) or any(not isinstance(v, str) or not v.strip() or '\n' in v or '\r' in v for v in bindings.values()):
            raise ValueError("bindings는 입력 슬롯과 일치하는 비어 있지 않은 한 줄 문자열이어야 합니다.")
        if any(set(fields(x)) - set(names) for x in outputs):
            raise ValueError("출력에 입력으로 확인할 수 없는 슬롯이 있습니다.")
        if template.format_map(bindings) != row["input"] or [x.format_map(bindings) for x in outputs] != records:
            raise ValueError("템플릿과 bindings로 Gold 입력·정답을 정확히 재현해야 합니다.")
    return row


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(validate(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError("중복 id가 있습니다.")
    return rows


def key(row):
    # No typo correction, whitespace folding or context inference: avoid false exact hits.
    return row["context"], row["input"]


class Model:
    def __init__(self, rows, automatic=True, *, classification_index=None):
        self.rows = [validate(row) for row in rows]
        self.classifications = classification_index
        if classification_index is not None:
            classification_index.bind(self.rows)
        self.exact = defaultdict(list)
        groups = defaultdict(list)
        for row in rows:
            self.exact[key(row)].append(row)
            p = row.get("pattern")
            if p:
                signature = (row["context"], p["input_template"], tuple(p["record_templates"]))
                groups[signature].append(row)
        self.patterns = []
        for (context, template, outputs), evidence in groups.items():
            # Duplicate imports or repeated identical examples do not create support.
            if len({r["input"] for r in evidence}) < 2:
                continue
            parts = []
            for literal, name, _, _ in Formatter().parse(template):
                parts.append(re.escape(literal))
                if name:
                    parts.append(f"(?P<{name}>[^\r\n]+?)")
            self.patterns.append((context, re.compile("".join(parts)), outputs, evidence))
        self.automatic_rules = discover(rows) if automatic else []
        if classification_index is not None:
            allowed = {r['id'] for r in rows
                       if classification_index.category(r) == 'general_transform'}
            # Preserve the existing discovery/conflict checks; only approved
            # general examples may support generalization to a new input.
            self.patterns = [p for p in self.patterns if all(r['id'] in allowed for r in p[3])]
            found = self.automatic_rules
            self.automatic_rules = [r for r in found if set(r.evidence_ids) <= allowed]
            self.classification_filtered_rules = len(found) - len(self.automatic_rules)
        encoded = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        self.revision = hashlib.sha256(encoded).hexdigest()

    def summary(self):
        active = sum(not rule.blocked_by for rule in self.automatic_rules)
        result = {"examples": len(self.rows), "active_patterns": len(self.patterns) + active,
                "automatic_candidates": len(self.automatic_rules), "automatic_active": active,
                "automatic_blocked": len(self.automatic_rules) - active, "revision": self.revision}
        if self.classifications is not None:
            result['classification_policy'] = {
                'generalization': 'reviewed_general_only',
                'filtered_numeric_rules': self.classification_filtered_rules,
                'set_triggers': 'exact_reviewed_input_context_only',
                'example_counts': {
                    kind: sum((self.classifications.category(r) or 'unclassified_local') == kind
                              for r in self.rows)
                    for kind in ('set_charting', 'general_transform', 'conditional_restore',
                                 'previous_record_reference', 'input_typo', 'unclassified_local')
                },
            }
        return result

    def transform(self, text, context="", allow_exact=True):
        result = {"route": "needs_review", "records": [], "evidence_ids": [],
                  "requires_review": True, "revision": self.revision}
        if self.classifications is not None:
            decision = self.classifications.lookup(text, context)
            result['classification'] = decision['classification']
            result['classification_status'] = decision['status']
            if decision['status'] in ('conflict', 'human_check_required'):
                return dict(result, evidence_ids=decision['evidence_ids'], reason=decision['reason'])
        if allow_exact and (matches := self.exact.get((context, text))):
            result["evidence_ids"] = [r["id"] for r in matches]
            if len({tuple(r["records"]) for r in matches}) > 1:
                return dict(result, reason="동일 입력·맥락에 상충하는 정답이 있습니다.")
            return dict(result, route="exact", records=matches[0]["records"], reason="확인된 동일 입력·맥락의 결과 재사용")
        candidates = []
        blocked = []
        for rule in self.automatic_rules:
            records = rule.generate(text, context)
            if records is not None:
                if rule.blocked_by:
                    blocked.extend(rule.evidence_ids + rule.blocked_by)
                else:
                    candidates.append((records, [{'id': identifier} for identifier in rule.evidence_ids]))
        for expected_context, regex, outputs, evidence in self.patterns:
            if expected_context != context:
                continue
            match = regex.fullmatch(text)
            if match and all(v.strip() for v in match.groupdict().values()):
                candidates.append(([x.format_map(match.groupdict()) for x in outputs], evidence))
        if blocked:
            return dict(result, evidence_ids=sorted(set(blocked)),
                        reason="자동 발견한 수치 패턴이 다른 학습 Gold의 정답과 상충하여 보류합니다.")
        if candidates:
            result["evidence_ids"] = sorted({r["id"] for _, examples in candidates for r in examples})
            if len({tuple(records) for records, _ in candidates}) == 1:
                return dict(result, route="learned", records=candidates[0][0], reason="서로 다른 Gold 2건 이상에서 지지된 수치 대응 패턴 또는 승인 템플릿 적용; 사용자 검토 필요")
            return dict(result, reason="여러 학습 패턴의 출력이 상충합니다.")
        return dict(result, reason="일치하는 학습 패턴이 없습니다. Gold 또는 사용자 확인이 필요합니다. 외부 LLM은 연결되지 않았습니다.")


def evaluate(training, cases, automatic=True):
    replay_model = Model(training, automatic=automatic)
    replay = []
    held_out = []
    for row in cases:
        prediction = replay_model.transform(row['input'], row['context'])
        replay.append({"id": row['id'], "expected": row['records'], "prediction": prediction,
                       "match": prediction['records'] == row['records']})
        # Exclude both ID and identical input/context to prevent memorization leakage.
        remaining = [r for r in training if r['id'] != row['id'] and key(r) != key(row)]
        prediction = Model(remaining, automatic=automatic).transform(row['input'], row['context'], allow_exact=False)
        held_out.append({"id": row['id'], "expected": row['records'], "prediction": prediction,
                         "excluded_examples": len(training) - len(remaining),
                         "match": prediction['records'] == row['records']})
    def report(items):
        return {"count": len(items), "matched": sum(x['match'] for x in items),
                "abstained": sum(x['prediction']['route'] == 'needs_review' for x in items), "results": items}
    return {"gold_reproduction": report(replay), "held_out_generalization": report(held_out),
            "notice": "기록 문자열과 순서의 완전 일치를 비교합니다. 임상적 정확성을 보장하지 않습니다."}

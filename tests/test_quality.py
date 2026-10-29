"""Clinical boundary regressions; deterministic playback is not model evaluation."""
from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from nursing_charting.classifications import runtime_index
from nursing_charting.llm import OpenAIClient, request_body
from nursing_charting.quality import normalize_draft, select_style_gold, UNIT_POLICY
from nursing_charting.review import draft, select_llm_gold

ROOT = Path(__file__).resolve().parents[1]
GOLD_BACKUP = ROOT/'backups/synthetic-training-gold-2026-10-22.json'
CONFLICT_REVIEWS = ROOT/'gold_conflict_reviews.json'
COMPARE_SCRIPT = ROOT/'scripts/compare_charting_quality.py'


def requires_local(path, reason):
    # The public repository excludes the full synthetic training Gold and local
    # review/comparison files; these tests run only in the private dev folder.
    return unittest.skipUnless(path.exists(), f'local-only {reason}: {path.name} not in public repo')


def value(record):
    return {'records': [record], 'uncertainties': [], 'omissions': [], 'evidence': []}


class QualityTests(unittest.TestCase):
    def test_ambiguous_slashes_preserve_numbers_units_and_wording(self):
        for text in ['HR 80/100', 'hr (80 / 100)', '심박수 80/100',
                     'HR 80에서 100/110', 'HR 80/100 회/min',
                     'SBP 120/140', 'SpO2 95/98', 'HR (80)/(100)',
                     'HR 80회/100', 'BP 120/80/60']:
            with self.subTest(text=text):
                before = value(text)
                saved = deepcopy(before)
                after, changes = normalize_draft(before, text)
                self.assertEqual(after['records'], [text])
                self.assertEqual(changes, [])
                self.assertTrue(any('슬래시' in q for q in after['uncertainties']))
                self.assertEqual(before, saved)
                self.assertEqual(normalize_draft(after, text)[0], after)

    def test_output_slash_cannot_reinterpret_supported_hr_range(self):
        after, changes = normalize_draft(value('HR 80/100'), 'HR 80-100')
        self.assertEqual(after['records'], ['HR 80/100'])
        self.assertFalse(changes)
        self.assertTrue(any('슬래시' in q for q in after['uncertainties']))

    def test_slash_guard_keeps_approved_bp_and_clear_hr_policy(self):
        text = 'BP 120/80, HR 80-100'
        after, _ = normalize_draft(value(text), text)
        self.assertEqual(after['records'], ['BP 120/80 mmHg, HR 80-100 회/min'])
        self.assertFalse(after['uncertainties'])
        text = 'HR 80/100, HR 82'
        after, _ = normalize_draft(value(text), text)
        self.assertEqual(after['records'], ['HR 80/100, HR 82 회/min'])

    def test_hr_transition_preserves_recovery_and_normalizes_rate(self):
        text = '침상 가장자리 앉을 때 HR 82에서 108까지 상승, 쉬고 나니 90대로 돌아옴.'
        before = value('침상 가장자리 앉을 때 HR 82회에서 108회까지 상승함. 휴식 후 90대로 회복됨.')
        saved = deepcopy(before)
        after, changes = normalize_draft(before, text)
        self.assertEqual(after['records'], ['침상 가장자리 앉을 때 HR 82 회/min에서 108 회/min까지 상승함. 휴식 후 90대로 회복됨.'])
        self.assertEqual(before, saved)
        self.assertEqual(changes[0]['basis'], ['C06-R02'])
        self.assertEqual(normalize_draft(after, text)[0], after)

    def test_unapproved_hemodynamic_and_lab_units_are_not_restored(self):
        text = 'PA catheter 눈금 48cm 고정 확인. CVP 8, PA 34/16, 숫자 변동은 크지 않음. K 3.2'
        result, changes = normalize_draft(value(text), text)
        self.assertEqual(result['records'], [text])
        self.assertEqual(changes, [])
        self.assertTrue(all(any(label in q for q in result['uncertainties']) for label in ['CVP', 'PA', 'K']))

    @requires_local(GOLD_BACKUP, 'training Gold')
    def test_only_approved_measurement_notation(self):
        text = 'sbp 102, mbp 70, bp 110/60, spo2 95, HR 82'
        after, _ = normalize_draft(value(text), text)
        self.assertEqual(after['records'], ['SBP 102 mmHg, MBP 70 mmHg, BP 110/60 mmHg, SpO₂ 95 %, HR 82 회/min'])
        self.assertFalse(after['uncertainties'])
        rows = json.loads((ROOT/'backups/synthetic-training-gold-2026-10-22.json').read_text())['rows']
        index = runtime_index(rows)
        self.assertEqual(index.summary()['classification_counts'], {'conditional_restore':12, 'general_transform':81,
                         'input_typo':1, 'previous_record_reference':9, 'set_charting':24})
        for rule in UNIT_POLICY.values():
            for identifier in rule['basis']:
                self.assertEqual(index.labels[identifier]['classification'], 'general_transform')

    def test_dose_route_and_ambiguous_values_are_never_repaired(self):
        for text in ['놀핀 0.05, K 3.2, RR 20, MAP 70', 'HR 82,5', 'HR 90대', 'HR 82 mg']:
            with self.subTest(text=text):
                after, _ = normalize_draft(value(text), text)
                self.assertEqual(after['records'], [text])
        after, _ = normalize_draft(value('HR 82 mmHg'), 'HR 82')
        self.assertEqual(after['records'], ['HR 82 mmHg'])
        self.assertTrue(any('상충' in q for q in after['uncertainties']))

    def test_additional_and_missing_numbers_are_flagged_not_repaired(self):
        after, _ = normalize_draft(value('HR 108'), 'HR 82')
        self.assertEqual(after['records'], ['HR 108'])
        self.assertTrue(any('추가' in q for q in after['uncertainties']))
        self.assertTrue(any('누락' in q for q in after['uncertainties']))

    def test_relevance_is_source_based_capped_and_does_not_pad(self):
        def row(i, source, target):
            return {'id':i, 'input':source, 'context':'', 'records':[target], 'synthetic':True}
        rows = [row(str(i), 'HR 80 관찰', '승인 기록') for i in range(20)]
        unrelated = row('unrelated', 'nexium 8 mg/hr', 'HR 80')
        selected = select_style_gold(rows + [unrelated], 'HR 82')
        self.assertEqual(len(selected),12)
        self.assertNotIn(unrelated,selected)
        self.assertEqual(select_style_gold(rows, '기침 없다고 함'), [])
        self.assertEqual(select_style_gold(list(reversed(rows)), 'HR 82'),selected)

    @requires_local(GOLD_BACKUP, 'training Gold')
    def test_all_gold_safety_precedes_selection_and_global_notes_are_diagnostics(self):
        rows = json.loads((ROOT/'backups/synthetic-training-gold-2026-10-22.json').read_text())['rows']
        before = deepcopy(rows)
        class Client:
            calls = []
            def generate(self, text, context, gold, **kwargs):
                self.sent = gold
                return value('삽입부위 출혈 없음.')
        client = Client()
        with patch('urllib.request.build_opener', side_effect=AssertionError('network forbidden')):
            session = draft(rows, 'A-line 삽입부위 출혈 안보임', '', client, synthetic=True)
        snap = session['snapshot']
        self.assertTrue(snap['gold_diagnostics']['review_notes'])
        self.assertFalse(any('C04-R04' in q for q in snap['prediction']['uncertainties']))
        self.assertTrue(snap['prediction']['workflow_notices'])
        self.assertEqual(rows,before)
        self.assertLessEqual(len(client.sent),12)
        self.assertEqual(len(snap['gold_classifications']['excluded_from_llm']),22)
        self.assertNotIn('C06-R07',[r['id'] for r in client.sent])

    def test_synthetic_prompt_is_short_coherent_and_has_no_answers(self):
        client = OpenAIClient()
        captured = []
        def request(body, validator, **kwargs):
            captured.append(body)
            return validator({'inputs':['HR82 앉으면108 쉬고90대']})
        with patch.object(client, '_request', side_effect=request):
            self.assertEqual(len(client.generate_synthetic(20, ['피해야 할 원문'])),1)
        system = captured[0]['messages'][0]['content']
        for phrase in ['fragments', 'typos', 'planned versus completed', 'incompatible', 'never label or authorize']:
            self.assertIn(phrase,system)
        self.assertEqual(set(json.loads(captured[0]['messages'][1]['content'])),{'count','avoid_inputs'})
        prompt = request_body('K 3.2', '', [], 'mock')['messages'][0]['content']
        self.assertIn('not a paraphrase',prompt)
        self.assertIn('no relevant Gold',prompt)
        self.assertIn('Deterministic approved notation',prompt)

    @requires_local(CONFLICT_REVIEWS, 'Gold conflict review')
    def test_legacy_queue_management_notes_move_only_in_view(self):
        from nursing_charting.review import draft_view, fingerprint
        note = json.loads((ROOT/'gold_conflict_reviews.json').read_text())['reviews'][0]['review_note']
        session = {'snapshot':{'origin':{'id':'legacy'}, 'prediction':{
            'route':'llm','llm_attempted':True,'records':['원본 출력'],
            'uncertainties':[note,'현재 환자의 단위 확인 필요'],'omissions':[]},
            'gold_ids_sent':[], 'excluded_conflicting_ids':['C04-R04'], 'api_calls':[],
            'gold_conflict_reviews':{'applied':[],'invalidated':[]}}}
        before = deepcopy(session)
        digest = fingerprint(session['snapshot'])
        view = draft_view(session,127)
        self.assertEqual(view['uncertainties'],['현재 환자의 단위 확인 필요'])
        self.assertIn(note,view['diagnostics']['review_notes'])
        self.assertEqual(session,before)
        self.assertEqual(fingerprint(session['snapshot']),digest)

    @requires_local(COMPARE_SCRIPT, 'offline comparison script')
    def test_saved_queue_offline_comparison_preserves_all_bytes_and_review(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location('offline_comparison', ROOT/'scripts/compare_charting_quality.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # A structural test fixture, not the missing real 20-case review queue.
        job = {'items':[{'id':'fixture', 'input':'HR 82', 'session':{
            'snapshot':{'origin':{'context':''}, 'prediction':value('HR 82회'),
                        'api_calls':[{'usage':{'prompt_tokens':100,'completion_tokens':10}}]},
            'review':{'records':['수정 기록 보존'], 'notes':'검토 메모 보존'}}}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json'
            path.write_text(json.dumps(job,ensure_ascii=False),encoding='utf-8')
            before = path.read_bytes()
            with patch('urllib.request.build_opener',side_effect=AssertionError('network forbidden')):
                result = module.compare(path)
            self.assertEqual(path.read_bytes(),before)
        self.assertTrue(result['queue_present'])
        self.assertEqual(result['actual_provider_calls'],0)
        self.assertIsNone(result['cases'][0]['actual_model_after'])
        self.assertEqual(result['cases'][0]['saved_usage'],[{'prompt_tokens':100,'completion_tokens':10}])
        self.assertEqual(result['record_change_summary']['notation_changed'], ['fixture'])
        self.assertEqual(result['record_change_summary']['unchanged'], [])
        with tempfile.TemporaryDirectory() as directory:
            missing = module.compare(Path(directory)/'missing.json', require_queue=True)
        self.assertEqual(missing['source'], 'missing_requested_queue')
        self.assertEqual(missing['cases'], [])
        self.assertIsNone(missing['record_change_summary'])

    @requires_local(COMPARE_SCRIPT, 'offline comparison script')
    def test_queue_summary_separates_changed_unchanged_and_missing_drafts(self):
        import importlib.util
        import tempfile
        spec = importlib.util.spec_from_file_location('offline_comparison', ROOT/'scripts/compare_charting_quality.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # These are test fixtures only, not the user's unavailable 20 cases.
        items = []
        for identifier, text in [('changed','HR 82'), ('unchanged','HR 80/100')]:
            items.append({'id':identifier, 'input':text, 'session':{'snapshot':{
                'origin':{'context':''}, 'prediction':value(text), 'api_calls':[],
                'gold_ids_sent':['example-'+str(i) for i in range(105)]}}})
        items.append({'id':'no-draft', 'input':'검토 전', 'session':None})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json'
            path.write_text(json.dumps({'items':items},ensure_ascii=False),encoding='utf-8')
            before = path.read_bytes()
            result = module.compare(path, require_queue=True)
            self.assertEqual(path.read_bytes(),before)
        self.assertEqual(result['record_change_summary'], {'notation_changed':['changed'],
                         'unchanged':['unchanged'], 'no_saved_prediction':['no-draft']})
        self.assertEqual(result['cases'][0]['saved_gold_count'],105)
        self.assertEqual(result['cases'][0]['before_gold_count'],100)

    def test_explicit_pending_units_are_preserved_and_new_ones_are_excluded(self):
        for text in ['K 3.2 mmol/L', 'CVP 8 mmHg', 'RR 20 회/min']:
            after, changes = normalize_draft(value(text),text)
            self.assertEqual(after['records'],[text])
            self.assertFalse(after['uncertainties'])
            self.assertFalse(changes)
        after, changes = normalize_draft(value('K 3.2 mEq/L 확인됨.'),'K 3.2 확인됨.')
        self.assertEqual(after['records'],['K 3.2 확인됨.'])
        self.assertTrue(any('원문에 없는 단위' in q for q in after['uncertainties']))
        self.assertEqual(changes[0]['action'],'removed_unconfirmed_unit')

    def test_stated_unit_omission_is_flagged_and_korean_hr_label_is_standardized(self):
        after, _ = normalize_draft(value('K 3.2 확인됨.'), 'K 3.2 mmol/L 확인됨.')
        self.assertTrue(any('출력 누락' in q for q in after['uncertainties']))
        after, _ = normalize_draft(value('심박수 82회 측정됨.'),'HR 82 측정')
        self.assertEqual(after['records'],['HR 82 회/min 측정됨.'])

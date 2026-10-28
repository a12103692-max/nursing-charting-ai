"""Small, sequential synthetic review queue; JSON persistence, no scoring/engine."""

from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import threading
import unicodedata
from uuid import uuid4

from .cli import gold_lock, save, save_session
from .core import load_jsonl
from .classifications import runtime_index
from .llm import LLMError, OpenAIClient
from .review import approve, draft, draft_view


class CallLimit(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat()


def unique_inputs(inputs, avoid):
    def normalize(text):
        return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).casefold()
    known = [normalize(s) for s in avoid]
    result = []
    for text in inputs:
        value = normalize(text)
        if any(value == other or SequenceMatcher(None, value, other, autojunk=False).ratio() >= .92
               for other in known):
            continue
        result.append(text.strip())
        known.append(value)
    return result


class BudgetClient:
    """Reserve before dispatch, including interrupted/unknown calls in the cap."""
    def __init__(self, queue, job_id, item_id=None):
        self.queue, self.job_id, self.item_id = queue, job_id, item_id
        self.client = queue.client_factory()
        self.client.model = 'gpt-6-luna'

    @property
    def calls(self):
        return self.client.calls

    def _call(self, purpose, function, *args, **kwargs):
        identifier = self.queue.reserve(self.job_id, self.item_id, purpose)
        before = len(self.calls)
        try:
            return function(*args, **kwargs)
        finally:
            audit = self.calls[-1] if len(self.calls) > before else None
            self.queue.finish_call(self.job_id, identifier, audit)

    def generate(self, *args, **kwargs):
        return self._call('charting', self.client.generate, *args, **kwargs)

    def generate_synthetic(self, *args):
        return self._call('synthetic_inputs', self.client.generate_synthetic, *args)


class ReviewQueue:
    def __init__(self, store, *, client_factory=None):
        self.store = Path(store)
        self.root = self.store.parent / (self.store.stem + '-review-queue')
        self.root.mkdir(parents=True, exist_ok=True)
        self.guard = threading.RLock()
        self.worker = None
        self.closing = threading.Event()
        self.worker_error = None
        self.client_factory = client_factory or (lambda: OpenAIClient(conventions=True))
        try:
            self.recover()
        except (OSError, ValueError) as exc:
            # A damaged queue must not disable the original single-input UI.
            self.worker_error = type(exc).__name__

    def locked(self):
        return gold_lock(self.root / 'state')

    def lease(self):
        stack = ExitStack()
        try:
            stack.enter_context(gold_lock(self.root / 'worker', blocking=False))
        except OSError:
            stack.close()
            raise ValueError('다른 순차 작업이 실행 중입니다. 현재 작업을 중단한 뒤 다시 시도하세요.') from None
        return stack

    def path(self, identifier):
        if not isinstance(identifier, str) or not re.fullmatch('[a-f0-9]{32}', identifier):
            raise ValueError('올바르지 않은 대기열 ID입니다.')
        return self.root / (identifier + '.json')

    def read(self, identifier):
        try:
            job = json.loads(self.path(identifier).read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            raise ValueError('대기열 JSON이 손상되었습니다. 기존 파일을 초기화하지 않았습니다.') from None
        if (not isinstance(job, dict) or job.get('version') != 1 or job.get('id') != identifier or
                not {'items', 'calls', 'status', 'phase', 'call_limit', 'requested_count',
                     'pause_requested', 'created_at'}.issubset(job) or
                not isinstance(job['items'], list) or not isinstance(job['calls'], list) or
                any(not isinstance(i, dict) or not {'id','input','status','attempts','session'}.issubset(i)
                    for i in job['items']) or
                any(not isinstance(c, dict) or not {'id','status','purpose','item_id'}.issubset(c)
                    for c in job['calls'])):
            raise ValueError('대기열 형식을 확인하세요. 기존 파일을 초기화하지 않았습니다.')
        return job

    def write(self, job, *, create=False):
        job['updated_at'] = now()
        save_session(self.path(job['id']), job, create=create)

    def rows(self):
        with gold_lock(self.store):
            return load_jsonl(self.store)

    def recover(self):
        try:
            lease = self.lease()
        except ValueError:
            return  # Another process owns an active request; never steal its work.
        with lease, self.guard, self.locked():
            for path in self.root.glob('*.json'):
                job = self.read(path.stem)
                changed = False
                for call in job['calls']:
                    if call['status'] == 'in_flight':
                        call['status'] = 'interrupted_unknown'
                        changed = True
                for item in job['items']:
                    if item['status'] == 'processing':
                        item['status'] = 'interrupted'
                        item['error'] = 'interrupted_unknown: 호출 결과 미확인. 자동 재시도하지 않습니다.'
                        changed = True
                    if item.get('approval_intent') and item['status'] != 'approved':
                        changed = self.reconcile(item) or changed
                if job['status'] in ('running', 'pausing'):
                    job['status'] = 'paused'
                    job['error'] = '서버 재시작으로 중단되었습니다. 미처리 항목만 명시적으로 재개하세요.'
                    changed = True
                if changed:
                    self.write(job)

    def reconcile(self, item):
        """Recover Gold-written/queue-not-written approval without writing Gold."""
        intent = item['approval_intent']
        existing = next((r for r in self.rows() if r['id'] == intent['gold_id']), None)
        if existing is None:
            return False
        origin = item['session']['snapshot']['origin']
        if (all(existing.get(k) == v for k, v in origin.items()) and
                existing['records'] == intent['records'] and
                existing.get('review', {}).get('draft_sha256') == item['session']['snapshot_sha256']):
            item['status'] = 'approved'
            item['gold_id'] = existing['id']
            item['session']['status'] = 'approved'
        else:
            item['status'] = 'failed'
            item['error'] = 'approval_id_conflict: 저장된 ID가 다릅니다. 기존 Gold를 보존했습니다.'
        return True

    def summary(self, job):
        remaining = sum(i['status'] in ('waiting', 'processing') for i in job['items'])
        usage = {'input_tokens': 0, 'output_tokens': 0, 'reasoning_tokens': 0,
                 'unknown_calls': 0, 'unknown_reasoning_calls': 0}
        for call in job['calls']:
            value = call.get('audit', {}).get('usage')
            if (not isinstance(value, dict) or any(type(value.get(k)) is not int or value[k] < 0
                                                   for k in ('prompt_tokens', 'completion_tokens'))):
                usage['unknown_calls'] += 1
                usage['unknown_reasoning_calls'] += 1
                continue
            usage['input_tokens'] += value.get('prompt_tokens', 0) or 0
            usage['output_tokens'] += value.get('completion_tokens', 0) or 0
            details = value.get('completion_tokens_details')
            reasoning = details.get('reasoning_tokens') if isinstance(details, dict) else None
            if type(reasoning) is int and reasoning >= 0:
                usage['reasoning_tokens'] += reasoning
            else:
                usage['unknown_reasoning_calls'] += 1
        return {k: deepcopy(job.get(k)) for k in ('id', 'status', 'phase', 'requested_count',
                'created_at', 'updated_at', 'call_limit', 'error', 'duplicates_removed')} | {
            'total': len(job['items']), 'remaining': remaining, 'processed': len(job['items']) - remaining,
            'approved': sum(i['status'] == 'approved' for i in job['items']),
            'calls_reserved': len(job['calls']),
            'calls_attempted': sum(bool(c.get('audit')) for c in job['calls']), 'usage': usage}

    def list_jobs(self):
        self.recover()
        with self.guard, self.locked():
            jobs = [self.summary(self.read(p.stem)) for p in self.root.glob('*.json')]
            return {'jobs': sorted(jobs, key=lambda j: j['created_at'], reverse=True),
                    'worker_error': self.worker_error}

    def get(self, identifier):
        self.recover()
        with self.guard, self.locked():
            job = self.read(identifier)
            count = len(self.rows())
            items = []
            for item in job['items']:
                session = item.get('session')
                items.append({k: deepcopy(item.get(k)) for k in ('id', 'input', 'status', 'error', 'gold_id')} | {
                    'attempt_count': len(item['attempts']),
                    'view': draft_view(session, count) if session else None,
                    'review': deepcopy(session['review']) if session else None})
            return self.summary(job) | {'items': items, 'calls': deepcopy(job['calls']), 'gold_count': count}

    def reserve(self, job_id, item_id, purpose):
        with self.guard, self.locked():
            job = self.read(job_id)
            if len(job['calls']) >= job['call_limit']:
                raise CallLimit('call_limit_reached')
            identifier = uuid4().hex
            job['calls'].append({'id': identifier, 'item_id': item_id, 'purpose': purpose,
                                 'status': 'in_flight', 'started_at': now()})
            self.write(job)
            return identifier

    def finish_call(self, job_id, identifier, audit):
        with self.guard, self.locked():
            job = self.read(job_id)
            entry = next(c for c in job['calls'] if c['id'] == identifier)
            entry['status'] = 'finished' if audit else 'local_error'
            entry['finished_at'] = now()
            if audit:
                entry['audit'] = {k: deepcopy(audit.get(k)) for k in (
                    'requested_model', 'response_model', 'usage', 'http_status', 'status',
                    'prompt_sha256', 'elapsed_seconds')}
            self.write(job)

    def launch(self, job, lease):
        self.worker_error = None
        self.worker = threading.Thread(target=self.run, args=(job['id'], lease), daemon=True)
        try:
            self.worker.start()
        except BaseException:
            lease.close()
            raise

    def run(self, identifier, lease):
        try:
            self._run(identifier)
        except Exception as exc:
            # A disk/validation failure leaves the previous atomic file intact.
            self.worker_error = type(exc).__name__
        finally:
            lease.close()

    def _run(self, identifier):
        with self.guard, self.locked():
            job = self.read(identifier)
            generation = job['phase'] == 'generation' and not job['calls']
        if generation:
            try:
                rows = self.rows()
                avoid = [r['input'] for r in rows]
                with self.guard, self.locked():
                    for path in sorted(self.root.glob('*.json'))[-10:]:
                        avoid.extend(i['input'] for i in self.read(path.stem)['items'])
                avoid = avoid[-1000:]
                inputs = BudgetClient(self, identifier).generate_synthetic(job['requested_count'], avoid)
                # Gold answers are compared locally only, never sent to the generator.
                kept = unique_inputs(inputs, avoid + [s for r in rows for s in r['records']])
                with self.guard, self.locked():
                    job = self.read(identifier)
                    job['items'] = [{'id': uuid4().hex, 'input': text, 'status': 'waiting',
                                     'session': None, 'attempts': [], 'error': ''} for text in kept]
                    job['duplicates_removed'] = len(inputs) - len(kept)
                    job['generation_returned_count'] = len(inputs)
                    job['phase'] = 'charting'
                    if not kept:
                        job['status'] = 'failed'
                        job['error'] = '중복 제거 후 처리할 입력이 없습니다. 자동 추가 생성하지 않습니다.'
                    self.write(job)
                    if not kept:
                        return
            except (LLMError, ValueError, OSError, CallLimit) as exc:
                with self.guard, self.locked():
                    job = self.read(identifier)
                    job['status'] = 'failed'
                    job['error'] = self.error(exc) + ': 입력 생성 실패. 자동 재시도하지 않습니다. 새 배치를 명시적으로 시작하세요.'
                    self.write(job)
                return
        while True:
            with self.guard, self.locked():
                job = self.read(identifier)
                if job['pause_requested'] or self.closing.is_set():
                    job['status'] = 'paused'
                    self.write(job)
                    return
                item = next((i for i in job['items'] if i['status'] == 'waiting'
                             and (not job.get('only_item') or i['id'] == job['only_item'])), None)
                if item is None:
                    job['status'] = ('paused' if any(i['status'] == 'waiting' for i in job['items']) else
                                     'completed' if job['phase'] == 'charting' else 'failed')
                    job.pop('only_item', None)
                    self.write(job)
                    return
                item_id, text = item['id'], item['input']
                item['status'] = 'processing'
                item['attempts'].append({'started_at': now()})
                self.write(job)
            result, error = None, None
            try:
                result = draft(self.rows(), text, '', BudgetClient(self, identifier, item_id), synthetic=True)
            except (LLMError, ValueError, OSError, CallLimit) as exc:
                error = exc
            with self.guard, self.locked():
                job = self.read(identifier)
                item = next(i for i in job['items'] if i['id'] == item_id)
                item['attempts'][-1].update({'finished_at': now(), 'session': result,
                                             'error': self.error(error) if error else ''})
                if result:
                    item['session'] = result
                    item['status'] = ('held' if result['snapshot']['prediction']['route'] in
                                      ('conflict', 'needs_review') else 'ready')
                    item['error'] = ''
                else:
                    item['status'] = 'waiting' if isinstance(error, CallLimit) else 'failed'
                    item['error'] = self.error(error)
                    job['status'] = 'limit_reached' if isinstance(error, CallLimit) else 'paused'
                    job['error'] = item['error'] + ': 결과를 보존했습니다. 자동 재시도하지 않습니다.'
                self.write(job)
                if error:
                    return

    @staticmethod
    def error(exc):
        return str(exc) if isinstance(exc, (LLMError, CallLimit)) and re.fullmatch('[a-z_0-9]+', str(exc)) else type(exc).__name__

    def action(self, action, data):
        self.recover()
        if action not in ('start', 'pause', 'resume', 'edit', 'hold', 'skip', 'regenerate', 'approve'):
            raise ValueError('지원하지 않는 대기열 동작입니다.')
        if action in ('start', 'resume', 'regenerate') and data.get('cost_ack') is not True:
            raise ValueError('합성 전용 및 API 비용 발생 가능성을 확인하세요.')
        if action == 'start':
            count = data.get('count', 20)
            if type(count) is not int or not 1 <= count <= 50:
                raise ValueError('생성 건수는 1~50건이어야 합니다.')
        if action in ('start', 'resume'):
            limit = data.get('call_limit', count + 1 if action == 'start' else 21)
            if type(limit) is not int or not 1 <= limit <= 101:
                raise ValueError('API 호출 한도는 1~101회여야 합니다.')
        lease = self.lease() if action in ('start', 'resume', 'regenerate') else None
        try:
            with self.guard, self.locked():
                if action == 'start':
                    # Validate store before scheduling any paid request.
                    runtime_index(self.rows())
                    job = {'version': 1, 'id': uuid4().hex, 'created_at': now(), 'status': 'running',
                           'phase': 'generation', 'requested_count': count, 'call_limit': limit,
                           'pause_requested': False, 'items': [], 'calls': [], 'error': '', 'duplicates_removed': 0}
                    self.write(job, create=True)
                else:
                    job = self.read(data.get('job_id'))
                    if action == 'pause':
                        job['pause_requested'] = True
                        job['status'] = 'pausing' if job['status'] == 'running' else 'paused'
                    elif action == 'resume':
                        if limit < len(job['calls']):
                            raise ValueError('이미 예약한 호출 수보다 한도를 낮출 수 없습니다.')
                        if job['phase'] == 'generation':
                            raise ValueError('입력 생성 결과가 없습니다. 자동 재시도하지 않으므로 새 배치를 시작하세요.')
                        job.update(call_limit=limit, pause_requested=False, status='running', error='')
                        job.pop('only_item', None)
                    else:
                        item = next((i for i in job['items'] if i['id'] == data.get('item_id')), None)
                        if item is None:
                            raise ValueError('검토 항목이 없습니다.')
                        if item.get('approval_intent'):
                            self.reconcile(item)
                        if item['status'] == 'approved':
                            if action != 'approve':
                                raise ValueError('승인한 기록은 변경하거나 재생성할 수 없습니다.')
                        elif item['status'] == 'processing':
                            raise ValueError('처리 중입니다. 현재 호출이 끝난 뒤 검토하세요.')
                        elif action == 'regenerate':
                            if len(job['calls']) >= job['call_limit']:
                                raise ValueError('호출 한도에 도달했습니다. 한도를 명시적으로 늘린 뒤 재개하세요.')
                            if item.get('session'):
                                item.setdefault('review_history', []).append(deepcopy(item['session']))
                            item['status'] = 'waiting'
                            item['error'] = ''
                            item.pop('approval_intent', None)
                            job.update(pause_requested=False, status='running', error='', only_item=item['id'])
                        elif action in ('hold', 'skip'):
                            item['status'] = 'held' if action == 'hold' else 'skipped'
                        else:
                            session = item.get('session')
                            if session is None or session['snapshot']['prediction']['route'] == 'conflict':
                                raise ValueError('승인 가능한 초안이 없습니다. 충돌·실패 이유를 검토하세요.')
                            records, notes = data.get('records'), data.get('notes', '')
                            if (not isinstance(records, list) or len(records) != 1 or not isinstance(records[0], str)
                                    or (action == 'approve' and not records[0].strip()) or len(records[0]) > 20000 or
                                    not isinstance(notes, str) or len(notes) > 4000):
                                raise ValueError('단일 기록과 검토 메모를 확인하세요.')
                            session['review'] = {'records': records, 'notes': notes}
                            if action == 'approve':
                                if data.get('approved') is not True or not isinstance(data.get('reviewer'), str) or not data['reviewer'].strip():
                                    raise ValueError('검토 후 개별 승인과 승인자 이름이 필요합니다.')
                                item['approval_intent'] = {'gold_id': session['snapshot']['origin']['id'],
                                                          'records': deepcopy(records)}
                                self.write(job)  # Recovery intent precedes the Gold write.
                                with gold_lock(self.store):
                                    rows = load_jsonl(self.store)
                                    candidate = approve(rows, session, approved=True, reviewer=data['reviewer'])
                                    if candidate is None:
                                        item['status'] = 'held'
                                        item['error'] = 'gold_conflict: 기존 Gold와 충돌하여 저장하지 않았습니다.'
                                    else:
                                        save(self.store, rows + [candidate])
                                        item['status'] = 'approved'
                                        item['gold_id'] = candidate['id']
                self.write(job)
            if lease is not None:
                self.launch(job, lease)
                lease = None
            return self.get(job['id'])
        finally:
            if lease is not None:
                lease.close()

    def stop(self):
        self.closing.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2)

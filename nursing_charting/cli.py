import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from .core import Model, evaluate, load_jsonl
from .llm import LLMError, OpenAIClient, transform as llm_transform
from .llm_experiment import run as llm_experiment
from .review import draft as make_draft, approve as approve_draft
from .classifications import runtime_index


def emit(value):
    print(json.dumps(value, ensure_ascii=False, indent=2))


def save(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    Model(rows)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def gold_lock(path, *, blocking=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a+b') as stream:
        if os.name == 'nt':
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def save_session(path, value, *, create=False):
    # Exclusive creation keeps an earlier draft/baseline from being overwritten.
    if create:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description='합성 Gold 기반 간호기록 PoC (모든 결과 사용자 검토 필요)')
    parser.add_argument('--store', type=Path, default=Path('.nursing-data/gold.jsonl'))
    commands = parser.add_subparsers(dest='command', required=True)
    add = commands.add_parser('add', help='Gold 또는 검토 완료 수정 결과 등록')
    add.add_argument('file', type=Path)
    add.add_argument('--replace', action='store_true', help='동일 ID를 명시적으로 교체')
    commands.add_parser('learn', help='현재 데이터 재학습 및 활성 패턴 수 확인')
    draft = commands.add_parser('draft', help='Exact → Learned → Luna 초안; Gold에 자동 저장하지 않음')
    draft.add_argument('input')
    draft.add_argument('--context', default='')
    draft.add_argument('--synthetic', action='store_true')
    draft.add_argument('--output', type=Path, required=True, help='새 검토 JSON 파일')
    approve = commands.add_parser('approve', help='검토 JSON의 수정 기록을 명시적으로 승인·저장')
    approve.add_argument('file', type=Path)
    approve.add_argument('--approve', action='store_true')
    approve.add_argument('--reviewer', default='', help='승인자 식별명')
    transform = commands.add_parser('transform')
    transform.add_argument('input')
    transform.add_argument('--context', default='')
    transform.add_argument('--llm', action='store_true', help='Exact 다음 외부 LLM 사용; 기존 패턴 엔진 우회')
    transform.add_argument('--synthetic', action='store_true', help='현재 입력·맥락이 합성 데이터임을 명시')
    transform.add_argument('--without-gold', action='store_true', help='LLM에 학습 예시를 제공하지 않는 비교 조건')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--cases', type=Path, help='생략하면 등록 Gold 전체 평가')
    evaluation.add_argument('--output', type=Path)
    llm_evaluation = commands.add_parser('llm-evaluate', help='Exact 유지, 새 입력의 Gold 미제공/제공 LLM 비교')
    llm_evaluation.add_argument('--cases', type=Path, required=True)
    llm_evaluation.add_argument('--output', type=Path, required=True)
    llm_evaluation.add_argument('--max-calls', type=int, default=50)
    args = parser.parse_args()
    try:
        rows = load_jsonl(args.store) if args.store.exists() else []
        if args.command == 'add':
            incoming = load_jsonl(args.file)
            with gold_lock(args.store):
                rows = load_jsonl(args.store) if args.store.exists() else []
                ids = {r['id'] for r in incoming}
                if ids & {r['id'] for r in rows} and not args.replace:
                    raise ValueError('동일 ID가 이미 있습니다. 교체하려면 --replace를 사용하세요.')
                rows = [r for r in rows if r['id'] not in ids] + incoming
                save(args.store, rows)
            emit(dict(Model(rows).summary(), registered=len(incoming)))
        elif args.command == 'draft':
            if args.output.exists() or args.output.resolve() == args.store.resolve():
                raise ValueError('이전 출력/Gold와 다른 새 초안 경로를 지정하세요.')
            client = OpenAIClient(conventions=True)
            client.model = 'gpt-6-luna'
            result = make_draft(rows, args.input, args.context, client, synthetic=args.synthetic)
            save_session(args.output, result, create=True)
            emit(result)
        elif args.command == 'approve':
            if args.file.resolve() == args.store.resolve():
                raise ValueError('Gold 파일을 검토 파일로 사용할 수 없습니다.')
            with gold_lock(args.store):
                rows = load_jsonl(args.store) if args.store.exists() else []
                session = json.loads(args.file.read_text(encoding='utf-8'))
                candidate = approve_draft(rows, session, approved=args.approve, reviewer=args.reviewer)
                if candidate is not None:
                    save(args.store, rows + [candidate])
                save_session(args.file, session)
            emit({'status': session['status'], 'gold': candidate,
                  'conflict': session.get('conflict'),
                  'examples': len(rows) + (candidate is not None)})
            if candidate is None:
                return 2
        elif args.command == 'learn':
            emit(Model(rows, classification_index=runtime_index(rows)).summary())
        elif args.command == 'transform':
            if args.llm:
                emit(llm_transform(rows, args.input, args.context, OpenAIClient(),
                                   synthetic=args.synthetic, with_gold=not args.without_gold))
            else:
                if args.without_gold:
                    raise ValueError('--without-gold는 --llm과 함께 사용하세요.')
                emit(Model(rows, classification_index=runtime_index(rows)).transform(args.input, args.context))
        elif args.command == 'llm-evaluate':
            if args.output.resolve() in (args.store.resolve(), args.cases.resolve()):
                raise ValueError('평가 보고서로 데이터 파일을 덮어쓸 수 없습니다.')
            result = llm_experiment(rows, load_jsonl(args.cases), OpenAIClient(),
                                    args.output, args.max_calls)
            emit({k: result[k] for k in ('status', 'stop_reason', 'summary', 'api_measurement')})
            if result['status'] != 'completed':
                return 2
        else:
            result = evaluate(rows, load_jsonl(args.cases) if args.cases else rows)
            if args.output:
                if args.output.resolve() == args.store.resolve() or (args.cases and args.output.resolve() == args.cases.resolve()):
                    raise ValueError('평가 보고서로 데이터 파일을 덮어쓸 수 없습니다.')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            emit(result)
        return 0
    except (OSError, ValueError, TypeError, KeyError, LLMError) as exc:
        print(f'오류: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())

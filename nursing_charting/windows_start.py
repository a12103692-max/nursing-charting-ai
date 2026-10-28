"""Small Windows launcher; does not change the web, learning or review logic."""

import getpass
import http.client
import os
from pathlib import Path
import sys
import threading
import warnings
import webbrowser

from .backup import restore_gold, write_new
from .core import load_jsonl
from .web import create_server


KEY_NAME = 'NURSING_LLM_API_KEY'
BACKUP_NAME = 'public-demo-gold.json'


def valid_key(value):
    return bool(value) and not any(c.isspace() or ord(c) < 32 for c in value)


def read_key(path):
    """Read only the API-key field; never execute or expand .env content."""
    values = []
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:].strip()
        name, separator, value = line.partition('=')
        if name.strip() != KEY_NAME:
            continue
        if not separator:
            raise ValueError('기존 .env의 API 키 항목 형식을 확인하세요. 파일은 변경하지 않았습니다.')
        value = value.strip()
        if value.startswith(('"', "'")):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError('기존 .env의 따옴표 형식을 확인하세요. 파일은 변경하지 않았습니다.')
            value = value[1:-1]
        values.append(value)
    if len(values) != 1 or not valid_key(values[0]):
        raise ValueError('기존 .env에 유효한 NURSING_LLM_API_KEY 항목 하나가 필요합니다. 파일은 변경하지 않았습니다.')
    return values[0]


def prepare(root):
    env_file = root / '.env'
    if not env_file.exists():
        print('최초 설정: API 키를 숨김 입력하면 이 PC의 .env 파일에 저장합니다.')
        print('.env는 평문 비밀 파일입니다. GitHub나 다른 사람에게 공유하지 마세요.')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error', getpass.GetPassWarning)
                key = getpass.getpass('OpenAI API 키 (취소: 빈 입력 또는 Ctrl+C): ').strip()
        except (EOFError, getpass.GetPassWarning):
            raise ValueError('키를 숨겨 입력할 수 없는 터미널입니다. Windows 터미널에서 실행하세요.') from None
        if not valid_key(key):
            raise ValueError('설정을 취소했거나 키 형식이 올바르지 않습니다. .env를 만들지 않았습니다.')
        # Atomic exclusive publication: even a concurrently created file is preserved.
        write_new(env_file, f'{KEY_NAME}={key}\n')
        print('.env 설정을 저장했습니다. 키 값은 출력하지 않습니다.')
    key = read_key(env_file)
    store = root / '.nursing-data' / 'gold.jsonl'
    if not store.exists():
        restore_gold(root / 'examples' / BACKUP_NAME, store)
        print('공개용 합성 데모 예시를 새 저장 경로에 복원했습니다.')
    load_jsonl(store)
    return key, store


def run(root, port=8765):
    key, store = prepare(Path(root))
    previous = os.environ.get(KEY_NAME)
    os.environ[KEY_NAME] = key
    server = None
    thread = None
    try:
        server = create_server(store, port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        # Verify readiness locally before opening a browser. No LLM calls.
        connection = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=5)
        try:
            connection.request('GET', '/api/state')
            response = connection.getresponse()
            response.read()
            if response.status != 200:
                raise ValueError('Gold 상태를 확인하지 못했습니다. 데이터와 파일 권한을 확인하세요.')
        finally:
            connection.close()
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'합성 데이터 전용 웹 서버: {url} (종료: Ctrl+C)', flush=True)
        try:
            opened = webbrowser.open(url, new=2)
        except (webbrowser.Error, OSError):
            opened = False
        if not opened:
            print('브라우저를 자동으로 열지 못했습니다. 위 주소를 직접 열어주세요.')
        thread.join()
        return 0
    finally:
        if server is not None:
            if thread is not None and thread.is_alive():
                server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=5)
        if previous is None:
            os.environ.pop(KEY_NAME, None)
        else:
            os.environ[KEY_NAME] = previous


def main():
    if sys.version_info < (3, 12):
        print('Python 3.12 이상이 필요합니다.')
        return 1
    try:
        return run(Path(__file__).resolve().parent.parent)
    except KeyboardInterrupt:
        print('\n종료했습니다. 기존 .env와 Gold는 유지됩니다.')
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        # Never echo parsing exception contents, file lines or key values.
        print(f'시작 실패 ({type(exc).__name__}). .env의 키 항목, Gold/백업, 폴더 권한 또는 포트 사용 여부를 확인하세요.')
        print('기존 파일은 덮어쓰지 않습니다. WINDOWS.md를 참고하세요.')
        return 1


if __name__ == '__main__':
    sys.exit(main())

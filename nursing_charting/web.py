"""Loopback-only UI for the existing synthetic draft/review workflow."""

import argparse
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import secrets
import threading

from .cli import gold_lock, save
from .core import load_jsonl
from .llm import LLMError, OpenAIClient
from .review import approve, draft, draft_view
from .queue import ReviewQueue


STATIC = Path(__file__).with_name('static')


class RequestError(Exception):
    def __init__(self, status, message):
        self.status, self.message = status, message


class Application:
    def __init__(self, store):
        self.store = Path(store)
        self.csrf = secrets.token_urlsafe(32)
        self.sessions = {}
        self.lock = threading.Lock()
        self.queue = ReviewQueue(self.store)

    def rows(self):
        if not self.store.exists():
            raise ValueError('Gold 저장소가 없습니다. 합성 Gold 백업을 먼저 복원하세요.')
        return load_jsonl(self.store)

    def state(self):
        with gold_lock(self.store):
            return {'gold_count': len(self.rows()), 'csrf': self.csrf}

    def generate(self, data):
        text = data.get('input')
        if not isinstance(text, str) or not text.strip() or len(text) > 12000:
            raise ValueError('자연어 입력은 비어 있지 않은 12,000자 이내 문자열이어야 합니다.')
        if data.get('synthetic') is not True:
            raise ValueError('실제 환자정보가 없는 합성 입력임을 확인하세요.')
        if len(self.sessions) >= 64:
            raise ValueError('검토 중인 초안이 너무 많습니다. 새 입력 버튼으로 기존 초안을 정리하세요.')
        with gold_lock(self.store):
            rows = self.rows()
        client = OpenAIClient(conventions=True)
        client.model = 'gpt-6-luna'
        # Web input has no separate context, including requests from old clients.
        session = draft(rows, text, '', client, synthetic=True)
        identifier = session['snapshot']['origin']['id']
        self.sessions[identifier] = session
        return draft_view(session, len(rows))

    def accept(self, data):
        identifier = data.get('id')
        if not isinstance(identifier, str) or identifier not in self.sessions:
            raise ValueError('검토 초안을 찾을 수 없습니다. 서버 재시작 후에는 다시 생성하세요.')
        # Keep the pending draft usable if disk persistence fails.
        session = deepcopy(self.sessions[identifier])
        session['review'] = {'records': data.get('records'), 'notes': data.get('notes', '')}
        with gold_lock(self.store):
            rows = self.rows()
            candidate = approve(rows, session, approved=data.get('approved'), reviewer=data.get('reviewer'))
            if candidate is None:
                self.sessions[identifier] = session
                ids = ', '.join(session['conflict']['existing_ids'])
                raise RequestError(409, '기존 Gold와 충돌하여 저장하지 않았습니다. 검토할 ID: ' + ids)
            save(self.store, rows + [candidate])
            self.sessions[identifier] = session
        return {'status': 'approved', 'gold_count': len(rows) + 1, 'gold_id': candidate['id']}


def create_server(store, port=8765):
    app = Application(store)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            # Do not log synthetic input, request headers or credentials.
            pass

        def respond(self, status, body, content_type='application/json; charset=utf-8'):
            if isinstance(body, dict):
                body = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
            self.send_header('Referrer-Policy', 'no-referrer')
            self.end_headers()
            self.wfile.write(body)

        def validate_request(self, mutate=False):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get_all('Host') != [host]:
                raise RequestError(403, '허용되지 않은 로컬 주소입니다. 127.0.0.1 주소로 접속하세요.')
            origins = self.headers.get_all('Origin')
            if origins is not None and origins != ['http://' + host]:
                raise RequestError(403, '외부 출처 요청을 거부했습니다.')
            if self.headers.get('Sec-Fetch-Site') not in (None, 'none', 'same-origin'):
                raise RequestError(403, '외부 페이지 요청을 거부했습니다.')
            if mutate:
                if origins != ['http://' + host]:
                    raise RequestError(403, '동일 출처 확인이 필요합니다.')
                if not secrets.compare_digest(self.headers.get('X-CSRF-Token', ''), app.csrf):
                    raise RequestError(403, '요청 확인 토큰이 만료되었습니다. 페이지를 새로고침하세요.')

        def do_GET(self):
            try:
                self.validate_request()
                if self.path == '/api/state':
                    self.respond(200, app.state())
                    return
                if self.path == '/api/queue':
                    self.respond(200, app.queue.list_jobs())
                    return
                if self.path.startswith('/api/queue/'):
                    self.respond(200, app.queue.get(self.path[len('/api/queue/'):]))
                    return
                files = {'/': ('index.html', 'text/html; charset=utf-8'),
                         '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                         '/queue.js': ('queue.js', 'text/javascript; charset=utf-8'),
                         '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if self.path not in files:
                    raise RequestError(404, '페이지를 찾을 수 없습니다.')
                name, content_type = files[self.path]
                self.respond(200, (STATIC / name).read_bytes(), content_type)
            except RequestError as exc:
                self.respond(exc.status, {'error': exc.message})
            except (ValueError, OSError):
                self.respond(500, {'error': '검토 대기열을 읽지 못했습니다. 파일을 보존했습니다. JSON·권한을 확인하세요.'
                                  if self.path.startswith('/api/queue') else
                                  'Gold를 읽을 수 없습니다. 백업 복원과 파일 권한을 확인하세요.'})

        def do_POST(self):
            try:
                self.validate_request(mutate=True)
                if self.headers.get('Content-Type') != 'application/json':
                    raise RequestError(415, 'JSON 요청만 허용합니다.')
                if self.headers.get('Transfer-Encoding'):
                    raise RequestError(400, '지원하지 않는 요청 형식입니다.')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 100000:
                    raise RequestError(413, '요청 크기가 허용 범위를 벗어났습니다.')
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ValueError('요청은 JSON 객체여야 합니다.')
                with app.lock:
                    if self.path == '/api/draft':
                        result = app.generate(data)
                    elif self.path == '/api/approve':
                        result = app.accept(data)
                    elif self.path == '/api/discard':
                        identifier = data.get('id')
                        if isinstance(identifier, str):
                            app.sessions.pop(identifier, None)
                        result = {'status': 'discarded'}
                    elif self.path.startswith('/api/queue/'):
                        result = app.queue.action(self.path.rsplit('/', 1)[-1], data)
                    else:
                        raise RequestError(404, '요청을 찾을 수 없습니다.')
                self.respond(200, result)
            except RequestError as exc:
                self.respond(exc.status, {'error': exc.message})
            except LLMError as exc:
                self.respond(502, {'error': 'Luna 호출 실패: ' + str(exc) + '. 자동 재시도하지 않았습니다.'})
            except (ValueError, TypeError, KeyError) as exc:
                message = str(exc) if isinstance(exc, ValueError) and not isinstance(exc, json.JSONDecodeError) else '올바르지 않은 요청입니다.'
                self.respond(400, {'error': message})
            except OSError:
                self.respond(500, {'error': 'Gold 읽기/저장에 실패했습니다. 파일 권한과 잠금 상태를 확인하세요. 기존 Gold를 덮어쓰지 않았습니다.'})

    class LocalServer(ThreadingHTTPServer):
        def server_close(self):
            app.queue.stop()
            super().server_close()

    server = LocalServer(('127.0.0.1', port), Handler)
    server.app = app
    return server


def main():
    parser = argparse.ArgumentParser(description='합성 간호차팅 로컬 웹 UI')
    parser.add_argument('--store', type=Path, default=Path('.nursing-data/gold.jsonl'))
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        load_jsonl(args.store)
        server = create_server(args.store, args.port)
    except (OSError, ValueError):
        parser.exit(2, '시작 실패: Gold 복원 여부, 파일 권한 또는 포트 사용 여부를 확인하세요.\n')
    print(f'합성 데이터 전용: http://127.0.0.1:{server.server_port} (종료: Ctrl+C)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

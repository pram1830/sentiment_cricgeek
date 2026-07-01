import json
import urllib.request
import urllib.error
import time

BASE = 'http://127.0.0.1:8000'

def request(method, path, payload=None, headers=None):
    data = None
    req_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode()
        req_headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(BASE + path, data=data, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return resp.status, body
    except urllib.error.HTTPError as err:
        body = err.read().decode()
        return err.code, body
    except Exception as exc:
        return None, str(exc)

status, body = request('GET', '/health')
print('health', status, body)

unique = int(time.time())
payload = {'username': f'testuser{unique}', 'email': f'testuser{unique}@example.com', 'password': 'TestPassword123!'}
status, body = request('POST', '/api/auth/signup', payload)
print('signup', status, body)

if status in (200, 201):
    token_payload = json.loads(body)
    token = token_payload.get('access_token')
    headers = {'Authorization': f'Bearer {token}'} if token else {}
    status, body = request('GET', '/api/users/me', headers=headers)
    print('me', status, body)
    status, body = request('GET', '/api/score/history', headers=headers)
    print('history', status, body)
    status, body = request('GET', '/api/rate-limit/status', headers=headers)
    print('rate', status, body)
    status, body = request('POST', '/api/score', {'text': 'India won the final over with a strong finish and smart bowling plans.', 'model': 'eqs', 'community_id': None}, headers=headers)
    print('score', status, body)

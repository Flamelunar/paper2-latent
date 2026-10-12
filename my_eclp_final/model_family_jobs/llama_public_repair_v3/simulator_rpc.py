"""Local-only shared simulator using the existing per-row seeded batch sampler."""
import json
import urllib.request
import urllib.error
import http.client
import socket

class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=900):
        super().__init__('localhost', timeout=timeout)
        self.path=path
    def connect(self):
        self.sock=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)

def unix_request(path, route, payload=None, timeout=900):
    conn=UnixConnection(path, timeout)
    try:
        body=None if payload is None else json.dumps(payload,ensure_ascii=False).encode()
        conn.request('GET' if payload is None else 'POST',route,body=body,
                     headers={'Content-Type':'application/json'})
        response=conn.getresponse();data=response.read()
        if response.status!=200:raise RuntimeError(data.decode(errors='replace'))
        return json.loads(data)
    finally:conn.close()

class RemoteSimulator:
    def __init__(self, url):
        self.url = url.rstrip('/')
    def close(self):
        pass
    def complete_batch(self, requests):
        payload = [{'system': r.system, 'prompt': r.prompt, 'seed': r.seed,
                    'point': r.point} for r in requests]
        if self.url.startswith('unix:'):
            return [tuple(item) for item in unix_request(self.url[5:],'/generate',payload)]
        req = urllib.request.Request(self.url + '/generate',
              data=json.dumps(payload, ensure_ascii=False).encode(),
              headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                return [tuple(item) for item in json.load(response)]
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors='replace')
            raise RuntimeError('Shared simulator failed: ' + message) from exc

def install_remote_batching():
    import batching
    original = batching.complete_batch
    def complete_batch(requests):
        if isinstance(requests[0].endpoint, RemoteSimulator):
            assert all(r.endpoint is requests[0].endpoint for r in requests)
            return requests[0].endpoint.complete_batch(requests)
        return original(requests)
    batching.complete_batch = complete_batch

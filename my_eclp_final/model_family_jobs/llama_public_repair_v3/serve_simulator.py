"""Single shared BF16 simulator on GPU4; no Judge or actor metadata supplied."""
import argparse
import json
import os
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from socketserver import UnixStreamServer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--socket',type=Path,required=True)
    ap.add_argument('--status', type=Path, required=True)
    args = ap.parse_args()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '4'
    sys.path[:0] = [str(args.root/'my_eclp_final/seven_methods_192'),
                   str(args.root/'my_eclp_final'), str(args.root/'exp_code_ljj')]
    import torch
    from repro.simulation import LocalTransformersEndpoint
    from repro.gpu_capacity import wait_for_bf16_capacity
    from batching import Request, complete_batch
    from experiment_io import atomic_json
    model = '/app/hzj/liujianjian/Qwen3.5-9B-persuadee/merged'
    wait_for_bf16_capacity([model], [0], reserve_gib=4,
                          description='shared Llama192 simulator on physical GPU4')
    torch.cuda.set_per_process_memory_fraction(22*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
    endpoint = LocalTransformersEndpoint(model=model, device='cuda:0')
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b'{"ready":true}')
        def do_POST(self):
            try:
                if self.path != '/generate':
                    raise ValueError('Unknown endpoint')
                rows = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if not 1 <= len(rows) <= 2:
                    raise ValueError('Only one or two scenario requests per batch')
                requests = [Request(endpoint, r['system'], r['prompt'], r['seed'], r['point']) for r in rows]
                results = complete_batch(requests)
                body = json.dumps(results, ensure_ascii=False).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
            except Exception as exc:
                self.send_response(500); self.end_headers()
                self.wfile.write(json.dumps({'error': repr(exc)}).encode())
                raise
    if args.socket.exists():args.socket.unlink()
    server=UnixStreamServer(str(args.socket),Handler)
    args.socket.chmod(0o600)
    atomic_json(args.status, {'ready':True, 'pid':os.getpid(), 'socket':str(args.socket), 'gpu':4,
                             'model':model, 'decode':'existing RowSampler; independent request seeds'})
    print('Shared simulator ready', args.socket, flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.socket.exists():args.socket.unlink()
        endpoint.close()

if __name__ == '__main__':
    main()

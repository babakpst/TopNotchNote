"""inference_server.py
Companion example for: topnotchnote.com/gpu/inference_server.html

A minimal GPU inference server, standard library plus PyTorch:
  * an HTTP endpoint, POST /predict, body = 2,048 float32 values (raw bytes)
  * a request queue
  * one batcher thread that owns the GPU: it takes the first waiting
    request, then waits up to --max-delay-ms for more, up to --max-batch,
    runs ONE forward pass for the whole batch, and hands each request its row
  * a built-in load generator (--bench) that measures throughput and latency,
    with no batching, with batching, and with no model at all (to measure the
    Python HTTP front end on its own)

The model is a stand-in: four 2048 x 2048 linear layers with GELU, about
67 MB of float32 weights.

Run the server:          python inference_server.py --max-batch 32
Benchmark both settings: python inference_server.py --bench
"""
import argparse
import http.client
import json
import queue
import struct
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

DIM = 2048


def build_model(dev):
    torch.manual_seed(0)
    layers = []
    for _ in range(4):
        layers += [torch.nn.Linear(DIM, DIM), torch.nn.GELU()]
    return torch.nn.Sequential(*layers).to(dev).eval()


class Request:
    __slots__ = ("x", "done", "result")

    def __init__(self, x):
        self.x = x
        self.done = threading.Event()
        self.result = None


class Batcher:
    def __init__(self, model, dev, max_batch, max_delay_s):
        self.model, self.dev = model, dev
        self.max_batch, self.max_delay_s = max_batch, max_delay_s
        self.q = queue.Queue()
        self.batch_sizes = []
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, x):
        r = Request(x)
        self.q.put(r)
        r.done.wait()
        return r.result

    def _loop(self):
        while True:
            batch = [self.q.get()]                      # wait for the first one
            deadline = time.perf_counter() + self.max_delay_s
            while len(batch) < self.max_batch:
                left = deadline - time.perf_counter()
                if left <= 0:
                    break
                try:
                    batch.append(self.q.get(timeout=left))
                except queue.Empty:
                    break
            if self.model is None:                    # front end only, no GPU
                for r in batch:
                    r.result = {"argmax": 0, "value": 0.0}
                    r.done.set()
                self.batch_sizes.append(len(batch))
                continue
            x = torch.stack([r.x for r in batch]).pin_memory()
            with torch.no_grad():
                y = self.model(x.to(self.dev, non_blocking=True))
                top = y.max(dim=1)
                idx = top.indices.cpu().tolist()       # one sync per batch
                val = top.values.cpu().tolist()
            for r, i, v in zip(batch, idx, val):
                r.result = {"argmax": i, "value": round(v, 5)}
                r.done.set()
            self.batch_sizes.append(len(batch))


def make_handler(batcher):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"  # keep connections open between requests

        def do_POST(self):
            if self.path != "/predict":
                self.send_error(404)
                return
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if len(body) != 4 * DIM:
                self.send_error(400, f"expected {4 * DIM} bytes")
                return
            x = torch.frombuffer(bytearray(body), dtype=torch.float32)
            out = json.dumps(batcher.submit(x)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *args):
            pass

    return Handler


def serve(port, max_batch, max_delay_ms, dev, model):
    batcher = Batcher(model, dev, max_batch, max_delay_ms / 1000)
    ThreadingHTTPServer.request_queue_size = 256  # default backlog of 5 drops connections
    srv = ThreadingHTTPServer(("127.0.0.1", port), make_handler(batcher))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, batcher


def load(port, clients, per_client):
    payload = struct.pack(f"{DIM}f", *([0.01] * DIM))
    lat = []
    lock = threading.Lock()

    def client():
        conn = http.client.HTTPConnection("127.0.0.1", port)
        mine = []
        for _ in range(per_client):
            t0 = time.perf_counter()
            conn.request("POST", "/predict", body=payload,
                         headers={"Content-Type": "application/octet-stream"})
            resp = conn.getresponse()
            resp.read()
            assert resp.status == 200
            mine.append(time.perf_counter() - t0)
        with lock:
            lat.extend(mine)

    ts = [threading.Thread(target=client) for _ in range(clients)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    lat.sort()
    return len(lat) / wall, lat[len(lat) // 2] * 1e3, lat[int(len(lat) * 0.99)] * 1e3


def forward_ms(model, dev, b, reps=20):
    x = torch.randn(b, DIM, device=dev)
    with torch.no_grad():
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            model(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8077)
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--max-delay-ms", type=float, default=2.0)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda")
    model = build_model(dev)
    print(f"model: 4 x Linear({DIM}, {DIM}) + GELU on {torch.cuda.get_device_name(0)}")

    if not a.bench:
        serve(a.port, a.max_batch, a.max_delay_ms, dev, model)
        print(f"serving on http://127.0.0.1:{a.port}/predict  (Ctrl+C to stop)")
        threading.Event().wait()

    print("\nforward pass alone, by batch size:")
    for b in (1, 8, 32):
        ms = forward_ms(model, dev, b)
        print(f"   batch {b:>2}: {ms:6.2f} ms   ({b / ms * 1e3:7.0f} rows/s)")

    clients, per_client = 32, 40
    print(f"\nserver under load: {clients} concurrent clients x {per_client} requests")
    configs = ((a.port, 1, 0.0, model, ""), (a.port + 1, 32, 2.0, model, ""),
               (a.port + 2, 32, 2.0, None, "  <- no model: the HTTP front end alone"))
    for port, mb, delay, m, note in configs:
        srv, batcher = serve(port, mb, delay, dev, m)
        load(port, 4, 5)                     # warm-up
        batcher.batch_sizes.clear()
        rps, p50, p99 = load(port, clients, per_client)
        sizes = batcher.batch_sizes
        print(f"   max batch {mb:>2}, wait {delay:.0f} ms: {rps:6.0f} req/s, "
              f"latency p50 {p50:6.1f} ms, p99 {p99:6.1f} ms, "
              f"mean batch {sum(sizes) / len(sizes):4.1f}{note}")
        srv.shutdown()


if __name__ == "__main__":
    main()

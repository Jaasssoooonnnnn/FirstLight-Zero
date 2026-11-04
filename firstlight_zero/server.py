"""GPU inference service for the collectors: batches decision requests across engines, keeps LSTM states.

    python firstlight_zero/server.py --run RUN --port 31500 --device cuda:0

Protocol (TCP, 4-byte big-endian length + pickle), arrays batched over the request's n seats:
request = dict(client=int, n=int, model=[per seat 'current'|int], greedy=bool, key=int32[n], start=bool[n],
               inputs={unit_type int16[n,U], unit_feat f16[n,U,F], scal f16, cards f16, opp_cards f16, option_card int8, opt_feat f16},
               opt_mask bool[n,5], place_bits uint8[n,4,72], force bool[n]);
reply = dict(version=int, gate int8[n], option int8[n], cell int16[n], gate_free bool[n], logp f32[n], value f32[n]).
Recurrent state lives here, keyed by (client, key); `start` zeroes the row; an unknown key starts from zeros.
"""
from pathlib import Path
import argparse, json, pickle, socket, struct, sys, threading, time, traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import numpy as np
import torch

from firstlight_zero.model import XbowNet
from firstlight_zero.spec import HEIGHT, SCALARS, UNIT_FEATURES, WIDTH
from firstlight_zero.batch import FLOAT_KEYS, LONG_KEYS
from firstlight_zero.graph import GraphRunner, bucket

GRAPH_CACHE = 320      # max cached CUDA graphs per server process (one 16-seat graph per history snapshot + the popular ones)
GRAPH_MEM_FRACTION = .7   # of the device's memory for their private pools
# With many snapshots each history group must stay small (16 seats), or graphs stop fitting and the LRU recaptures constantly.


def recv_exact(sock, n):
    buf = bytearray(n); view = memoryview(buf); got = 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if not k:
            raise ConnectionError('closed')
        got += k
    return bytes(buf)


def recv_msg(sock):
    (n,) = struct.unpack('>I', recv_exact(sock, 4))
    return pickle.loads(recv_exact(sock, n))


def send_msg(sock, obj, lock=None):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    payload = struct.pack('>I', len(data)) + data
    if lock is not None:
        with lock: sock.sendall(payload)
    else:
        sock.sendall(payload)


class Models:
    def __init__(self, run, kwargs, device):
        self.run, self.kwargs, self.device = Path(run), kwargs, device
        self.current = None; self.version = -1; self.cache = {}
        self.refresh()

    def _load(self, path):
        m = XbowNet(**self.kwargs).to(self.device).eval()
        m.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        return m

    def refresh(self):
        try:
            latest = json.loads((self.run / 'weights/latest.json').read_text())
        except (OSError, ValueError):
            return
        if latest['version'] != self.version:
            self.current = self._load(self.run / 'weights' / latest['file']); self.version = latest['version']

    def get(self, key):
        if key == 'current':
            return self.current
        if key not in self.cache:
            # 5 MB per model: keep every snapshot PFSP can draw. (12 slots with 13+ snapshots reloaded a model from GPFS
            # on almost every batch -> 360 ms/batch, GPU idle
            if len(self.cache) >= 256:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key] = self._load(self.run / f'weights/history/v{key}.pt')
        return self.cache[key]


class Server:
    def __init__(self, run, port, device, settings, stop_file=None):
        self.device = torch.device(device); self.models = Models(run, settings['model'], self.device)
        self.pending = []; self.cv = threading.Condition(); self.port = port
        # LSTM state bank on the device: one row per (client, seat key); `start` zeroes the row before use.
        hidden = int(settings['model'].get('hidden', 256)); self.capacity = 32768
        self.bank = torch.zeros(2, self.capacity, hidden, device=self.device); self.rows = {}
        self.graphs = {}   # (model key, model version, bucket) -> GraphRunner; the whole forward is one CUDA graph replay
        self.use_graphs = self.device.type == 'cuda'; self.last_refresh = time.time(); self.reserved_gib = 0.
        self.graph_mem_gib = GRAPH_MEM_FRACTION * torch.cuda.get_device_properties(self.device).total_memory / 2**30 if self.use_graphs else 0.
        self.stats = dict(requests=0, seats=0, batches=0, gpu_s=0., wait_s=0., bad_requests=0, captures=0)
        self.max_wait = float(settings.get('server_batch_wait_ms', 3.)) / 1000.
        self.run = Path(run); self.stop_file = Path(stop_file) if stop_file else self.run / 'STOP'   # eval servers use their own flag

    def serve(self):
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('0.0.0.0', self.port)); srv.listen(512)
        threading.Thread(target=self.batcher, daemon=True).start()
        threading.Thread(target=self.housekeeping, daemon=True).start()
        while not self.stop_file.exists():
            srv.settimeout(2.)
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            threading.Thread(target=self.reader, args=(conn,), daemon=True).start()

    def reader(self, conn):
        lock = threading.Lock()
        try:
            while True:
                req = recv_msg(conn)
                with self.cv:
                    self.pending.append((conn, lock, req)); self.cv.notify()
        except (ConnectionError, OSError, EOFError):
            pass
        finally:
            conn.close()

    def housekeeping(self):
        last = time.time()
        while True:
            time.sleep(5.)
            if time.time() - last > 60:
                last = time.time()
                with (self.run / 'logs' / f'server-{self.port}.jsonl').open('a') as f:
                    f.write(json.dumps(dict(utc=time.time(), version=self.models.version, states=len(self.rows), graphs=len(self.graphs),
                                            eager=not self.use_graphs, reserved_gib=round(self.reserved_gib, 1), **self.stats)) + '\n')
                for k in self.stats: self.stats[k] = 0 if isinstance(self.stats[k], int) else 0.

    def batcher(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
                t0 = time.perf_counter(); deadline = t0 + self.max_wait
                while time.perf_counter() < deadline and sum(r[2]['n'] for r in self.pending) < 512:
                    self.cv.wait(timeout=max(0., deadline - time.perf_counter()))
                batch, self.pending = self.pending, []
            self.stats['wait_s'] += time.perf_counter() - t0
            if time.time() - self.last_refresh > 5.:        # all CUDA work, including weight loads, stays on this thread
                self.last_refresh = time.time()
                try: self.models.refresh()
                except Exception: traceback.print_exc(); sys.stdout.flush()
            batch = self.validate(batch)
            if not batch:
                continue
            try:
                self.process(batch)
            except Exception:
                traceback.print_exc(); sys.stdout.flush()
                for conn, lock, req in batch:
                    try: send_msg(conn, dict(error=traceback.format_exc()[-2000:]), lock)
                    except OSError: pass

    def validate(self, batch):
        """Reject malformed requests (e.g. a collector running an older input schema) one by one instead of failing the batch."""
        good = []
        for conn, lock, req in batch:
            try:
                inp = req['inputs']; n = int(req['n'])
                ok = (inp['unit_feat'].shape[-1] == UNIT_FEATURES and inp['scal'].shape == (n, SCALARS) and len(req['model']) == n
                      and req['place_bits'].shape == (n, 4, (HEIGHT * WIDTH + 7) // 8))
            except (KeyError, AttributeError, TypeError, ValueError):
                ok = False
            if ok:
                good.append((conn, lock, req)); continue
            self.stats['bad_requests'] += 1
            try: send_msg(conn, dict(error=f'malformed request: expected unit_feat width {UNIT_FEATURES}, scal width {SCALARS}'), lock)
            except OSError: pass
        return good

    def graph_runner(self, model, gk):
        """Cached CUDA-graph runner. Every graph owns a private memory pool proportional to its bucket (2.2 GiB at 1024, 1.1 at
        512, 0.55 at 256 — tools/graph_mem.py), so the cache is bounded by reserved device memory as well as by count: graphs
        of superseded 'current' versions are dropped as soon as a new version is served, then the least recently used go
        until the pools fit (freeing a runner releases its pool; empty_cache returns it to the device). A capture failure
        still falls back to eager execution for good rather than erroring every later batch."""
        runner = self.graphs.pop(gk, None)
        if runner is None:
            stale = [k for k in self.graphs if k[0] == 'current' and k[1] != gk[1]] if gk[0] == 'current' else []
            for k in stale:
                del self.graphs[k]
            if stale:
                torch.cuda.empty_cache()
            reserved = lambda: torch.cuda.memory_reserved(self.device) / 2**30
            while self.graphs and (len(self.graphs) >= GRAPH_CACHE or reserved() > self.graph_mem_gib):
                self.graphs.pop(next(iter(self.graphs))); torch.cuda.empty_cache()
            try:
                runner = GraphRunner(model, gk[2], self.device); self.stats['captures'] += 1
            except Exception:
                traceback.print_exc(); sys.stdout.flush()
                self.use_graphs = False; self.graphs.clear()
                print('CUDA graph capture failed; serving eagerly from now on', flush=True)
                return None
            self.reserved_gib = reserved()
        self.graphs[gk] = runner
        return runner

    def process(self, batch):
        t0 = time.perf_counter()
        reqs = [req for _, _, req in batch]
        counts = [int(r['n']) for r in reqs]; total = sum(counts)
        self.stats['requests'] += len(batch); self.stats['seats'] += total; self.stats['batches'] += 1
        cat = lambda get: np.concatenate([get(r) for r in reqs]) if len(reqs) > 1 else get(reqs[0])
        models = [m for r in reqs for m in r['model']]
        # Seats are sorted by model so every model group is a contiguous slice: inputs/legality go to the device once per
        # batch, only the forward (graph replay) runs per group, and sampling + the copy back happen once for the whole
        # batch. Per-group upload + sampling used to cost as much as the forward itself (tools/bench_groups.py: 18 groups
        # of 256 seats took 12.7x one group).
        groups = {}
        for j, m in enumerate(models):
            groups.setdefault(m, []).append(j)
        order = np.fromiter((j for g in groups.values() for j in g), np.int64, total)
        inverse = np.empty(total, np.int64); inverse[order] = np.arange(total)
        keys = cat(lambda r: r['key'])[order]; starts = cat(lambda r: r['start'])[order]
        clients = np.repeat([r['client'] for r in reqs], counts)[order]
        rows = np.empty(total, np.int64)
        for j in range(total):
            sk = (int(clients[j]), int(keys[j])); row = self.rows.get(sk)
            if row is None:
                if len(self.rows) >= self.capacity:
                    self.rows.clear(); self.bank.zero_()
                row = self.rows[sk] = len(self.rows)
            rows[j] = row
        dev = self.device
        x = {}
        for k in reqs[0]['inputs']:
            t = torch.from_numpy(np.ascontiguousarray(cat(lambda r, k=k: r['inputs'][k])[order])).to(dev)
            x[k] = t.long() if k in LONG_KEYS else t.float()
        place = np.unpackbits(cat(lambda r: r['place_bits'])[order], axis=-1)[..., :HEIGHT * WIDTH].reshape(total, 4, HEIGHT, WIDTH)
        legal = dict(opt_mask=torch.from_numpy(cat(lambda r: r['opt_mask'])[order]).to(dev), place=torch.from_numpy(place.astype(bool)).to(dev),
                     force=torch.from_numpy(cat(lambda r: r['force'])[order]).to(dev))
        idx = torch.from_numpy(rows).to(dev)
        if starts.any():
            self.bank[:, torch.from_numpy(rows[starts]).to(dev)] = 0.
        state = (self.bank[0, idx].unsqueeze(0).contiguous(), self.bank[1, idx].unsqueeze(0).contiguous())
        greedy = any(r.get('greedy') for r in reqs)
        version = self.models.version
        heads, h_state, c_state = [], [], []
        offset = 0
        for key, members in groups.items():
            n = len(members); sl = slice(offset, offset + n); offset += n
            model = self.models.get(key)
            xg = {k: v[sl] for k, v in x.items()}; lg = {k: v[sl] for k, v in legal.items()}; sg = (state[0][:, sl], state[1][:, sl])
            # history snapshots get coarse buckets so (snapshot x bucket) stays well inside the graph cache
            runner = self.graph_runner(model, (key, version if key == 'current' else key, bucket(n, coarse=key != 'current'))) if self.use_graphs else None
            if runner is not None:
                hg, sg = runner(xg, lg, sg)
            else:
                with torch.no_grad():
                    hg, sg = model.heads(xg, lg, sg)
            heads.append(hg); h_state.append(sg[0]); c_state.append(sg[1])
        if len(groups) > 1:
            heads = {k: torch.cat([h[k] for h in heads]) for k in heads[0]}
            new_state = (torch.cat(h_state, 1), torch.cat(c_state, 1))
        else:
            heads = heads[0]; new_state = (h_state[0], c_state[0])
        self.bank[0, idx] = new_state[0][0]; self.bank[1, idx] = new_state[1][0]
        if any('temp' in r for r in reqs):                    # eval requests: per-seat temperature (0 = argmax); training never sends it
            temp = torch.from_numpy(np.ascontiguousarray(cat(lambda r: r['temp'] if 'temp' in r else np.ones(int(r['n']), np.float32))[order]))
            res = model.sample_t(heads, legal, temp.to(dev).float())
        else:
            res = model.sample(heads, legal, greedy=greedy)   # sampling uses no weights: any XbowNet instance serves the whole batch
        casts = dict(gate=np.int8, option=np.int8, cell=np.int16, gate_free=bool, logp=np.float32, value=np.float32)
        out = {k: res[k].cpu().numpy()[inverse].astype(dt, copy=False) for k, dt in casts.items()}
        self.stats['gpu_s'] += time.perf_counter() - t0
        offset = 0
        for (conn, lock, req), n in zip(batch, counts):
            reply = dict(version=version, **{k: v[offset:offset + n] for k, v in out.items()}); offset += n
            try:
                send_msg(conn, reply, lock)
            except OSError:
                pass


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path, required=True); p.add_argument('--port', type=int, required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--stop-file', type=Path); a = p.parse_args()
    s = json.loads((a.run / 'settings.json').read_text())
    torch.backends.cudnn.benchmark = False   # batch sizes vary per request
    torch.distributions.Distribution.set_default_validate_args(False)
    torch.set_num_threads(2)                 # several server processes share the GPU node's cores
    server = Server(a.run, a.port, a.device, s, a.stop_file)
    host = socket.gethostname().split('.')[0]
    reg = a.run / 'servers'; reg.mkdir(exist_ok=True)
    (reg / f'{host}-{a.port}.json').write_text(json.dumps(dict(host=host, port=a.port, device=a.device, utc=time.time())))
    print('serving', host, a.port, a.device, flush=True)
    server.serve()


if __name__ == '__main__':
    main()

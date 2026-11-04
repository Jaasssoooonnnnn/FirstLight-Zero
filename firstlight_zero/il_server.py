"""GPU inference server for the IL-family opponent (a V4 imitation-learning checkpoint or its PPO post-training) in evaluations.

Eval workers tensorize the opponent seat locally (as before: `tz.tensorize`, `record_action` and decoding stay on the
worker), pad + concatenate their rows (`concatenate_padded_tensor_records`) and send one
request per decision round: per-row state keys, episode-start and forced-opening flags, and the batch with numpy arrays in
place of tensors (`to_wire`: pickling torch tensors costs ~100x more CPU and was the bottleneck at 60 workers per server).
The server concatenates the pending requests, runs `act` (argmax; earliest delay, 200 ms ceiling, asserted at load) with the recurrent
state kept in a device bank per (client, key), applies the forced-ACT path (`forward` + `act_after_preselected_act` on
the pre-inference state) for requested rows whose gate is WAIT and that have a legal candidate, and returns each row's
ActionSequenceV4 rows (one record per request, numpy) and whether each was forced.

    python firstlight_zero/il_server.py --checkpoint PT --registry DIR --port P --stop-file F [--device cuda:0]
"""
from pathlib import Path
import argparse, json, socket, sys, threading, time, traceback
from dataclasses import fields

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import torch  # noqa: E402

from firstlight_zero.server import send_msg, recv_msg  # noqa: E402

DELAYS_MS = [0, 50, 100, 150, 200]


def to_wire(record):
    """V4 tensor record -> the same record with numpy arrays (cheap to pickle)."""
    from native_runner.training.v4.tensors import _map_tensors
    return _map_tensors(record, lambda t: t.numpy())


def from_wire(record):
    import numpy as np
    from dataclasses import is_dataclass
    if isinstance(record, np.ndarray):
        return torch.from_numpy(record)
    if is_dataclass(record):
        return type(record)(**{f.name: from_wire(getattr(record, f.name)) for f in fields(record)})
    return record


class ILServer:
    def __init__(self, checkpoint, device, stop_file, max_wait_ms=3.):
        from firstlight_zero.opponent import load_opponent as load_il
        from native_runner.training.v4.tensors import concatenate_padded_tensor_records, ActionSequenceV4, GATE_ACT
        self.cat, self.fields, self.GATE_ACT = concatenate_padded_tensor_records, [f.name for f in fields(ActionSequenceV4)], GATE_ACT
        self.device = torch.device(device)
        self.il = load_il(Path(checkpoint), self.device, False)
        self.il.set_inference_delay_offset_max_ms(200)
        allowed = [int(v) for v, ok in zip(self.il.config.delay_offset_ms, self.il._inference_delay_offset_mask.cpu().tolist()) if ok]
        assert self.il.delay_policy == 'earliest' and not self.il.ppo_rollout_sampling and allowed == DELAYS_MS, (self.il.delay_policy, allowed)
        self.protocol = dict(mode='argmax', delay_policy=self.il.delay_policy, allowed_delay_offsets_ms=allowed, actual_delay_ceiling_ms=max(allowed),
                             runtime_cap_ms=self.il._inference_delay_offset_max_ms, device=str(self.device), checkpoint=str(checkpoint))
        self.capacity = 16384
        self.bank = self.il.initial_state(self.capacity, device=self.device); self.rows = {}
        self.pending = []; self.cv = threading.Condition(); self.stop_file = Path(stop_file); self.max_wait = max_wait_ms / 1000.
        self.stats = dict(requests=0, rows=0, batches=0, gpu_s=0., forced=0)

    def serve(self, port):
        srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); srv.bind(('0.0.0.0', port)); srv.listen(512)
        srv.settimeout(1.)
        threading.Thread(target=self.batcher, daemon=True).start()
        threading.Thread(target=self.housekeeping, daemon=True).start()
        while not self.stop_file.exists():
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
        except (OSError, EOFError, ConnectionError):
            pass

    def housekeeping(self):
        while not self.stop_file.exists():
            time.sleep(60)
            print(json.dumps(dict(utc=time.time(), rows_in_bank=len(self.rows), **self.stats)), flush=True)
            for k in self.stats: self.stats[k] = 0 if isinstance(self.stats[k], int) else 0.

    def batcher(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
            time.sleep(self.max_wait)           # let concurrent workers' requests join the batch
            with self.cv:
                batch, self.pending = self.pending, []
            try:
                self.process(batch)
            except Exception:
                err = traceback.format_exc(); print(err, flush=True)
                for conn, lock, _ in batch:
                    try: send_msg(conn, dict(error=err[-2000:]), lock)
                    except OSError: pass

    def process(self, batch):
        t0 = time.perf_counter(); dev = self.device
        rows = [(req['client'], key, start, force) for _, _, req in batch for key, start, force in zip(req['keys'], req['starts'], req['forces'])]
        idx = []
        for client, key, *_ in rows:
            r = self.rows.get((client, key))
            if r is None:
                if len(self.rows) >= self.capacity:
                    self.rows.clear()
                r = self.rows[(client, key)] = len(self.rows)
            idx.append(r)
        host = self.cat(tuple(from_wire(req['batch']) for _, _, req in batch))
        with torch.inference_mode():
            mb = host.to_model_input(dev)
            idx_t = torch.tensor(idx, dtype=torch.long, device=dev)
            state = self.bank.index_select(idx_t)
            start = torch.tensor([bool(row[2]) for row in rows], dtype=torch.bool, device=dev)
            out = self.il.act(mb, state, episode_start=start, validate=False)
            self.bank.hidden.index_copy_(0, idx_t, out.next_state.hidden.to(self.bank.hidden.dtype))
            self.bank.cell.index_copy_(0, idx_t, out.next_state.cell.to(self.bank.cell.dtype))
            actions = out.actions
            forced = torch.zeros(len(rows), dtype=torch.bool)
            want = [i for i, row in enumerate(rows) if row[3]]
            if want:
                w = torch.tensor(want, dtype=torch.long)
                legal = host.candidates.mask.index_select(0, w).flatten(1).any(1)
                wait = actions.gate.index_select(0, w.to(dev)).cpu() != self.GATE_ACT
                sel = w[legal & wait]
                if sel.numel():
                    sel_d = sel.to(dev)
                    fb = host.index_select(sel).to_model_input(dev)
                    ctx = self.il.forward(fb, state.index_select(sel_d), episode_start=start.index_select(0, sel_d), validate=False)
                    fo = self.il.act_after_preselected_act(fb, ctx, validate=False)
                    for name in self.fields:
                        getattr(actions, name).index_copy_(0, sel_d, getattr(fo.actions, name))
                    forced[sel] = True; self.stats['forced'] += int(sel.numel())
            actions = actions.to('cpu')
        self.stats['requests'] += len(batch); self.stats['rows'] += len(rows); self.stats['batches'] += 1
        self.stats['gpu_s'] += time.perf_counter() - t0
        k = 0
        for conn, lock, req in batch:
            n = len(req['keys'])
            reply = dict(actions=to_wire(actions.index_select(torch.arange(k, k + n))), forced=forced[k:k + n].tolist())
            k += n
            try:
                send_msg(conn, reply, lock)
            except OSError:
                pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True); p.add_argument('--registry', type=Path, required=True)
    p.add_argument('--port', type=int, required=True); p.add_argument('--stop-file', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    torch.set_num_threads(4)
    server = ILServer(a.checkpoint, a.device, a.stop_file)
    host = socket.gethostname().split('.')[0]; a.registry.mkdir(parents=True, exist_ok=True)
    (a.registry / f'{host}-{a.port}.json').write_text(json.dumps(dict(host=host, port=a.port, device=a.device, protocol=server.protocol, utc=time.time())))
    print('serving', host, a.port, a.device, json.dumps(server.protocol), flush=True)
    server.serve(a.port)


if __name__ == '__main__':
    main()

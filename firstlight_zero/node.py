"""On one engine host: run one collector per engine, restarting crashed ones."""
from pathlib import Path
import argparse, json, os, signal, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from firstlight_zero.hosts import load_engines   # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True); p.add_argument('--host', required=True)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args(); run = a.run
    s = json.loads((run / 'settings.json').read_text())
    res = load_engines(s['engines'])
    indices = [i for i, c in enumerate(res['collectors']) if c['host'] == a.host]
    if a.limit:
        indices = indices[:a.limit]
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
    logs = run / 'logs'; logs.mkdir(exist_ok=True)
    children = {}; restarts = {i: 0 for i in indices}; stopping = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopping.__setitem__(0, True))

    def spawn(i):
        out = (logs / f'collector-{i:03d}.log').open('a')
        children[i] = subprocess.Popen([sys.executable, str(ROOT / 'firstlight_zero/collector.py'), '--run', str(run), '--index', str(i)],
                                       env=env, stdout=out, stderr=subprocess.STDOUT)
        out.close()

    for i in indices:
        spawn(i)
    while not stopping[0] and not (run / 'STOP').exists():
        for i, proc in list(children.items()):
            if proc.poll() is not None and not (run / 'STOP').exists():
                restarts[i] += 1
                with (run / 'node-events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(event='collector_exit', index=i, code=proc.returncode, restarts=restarts[i], utc=time.time())) + '\n')
                if restarts[i] <= 20:
                    time.sleep(min(30, 2 * restarts[i])); spawn(i)
                else:
                    children.pop(i)
        time.sleep(2)
    for proc in children.values():
        if proc.poll() is None:
            proc.terminate()
    deadline = time.time() + 60
    for proc in children.values():
        try:
            proc.wait(max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == '__main__':
    main()

"""Start / stop GPU inference servers for evaluation (never while a training run uses the same run directory).

    python tools/eval_servers.py start RUN [--opponent OPP.pt] [--zero-gpus 0] [--opponent-gpus 0] > RUN/eval-servers.log 2>&1 &
    python tools/eval_servers.py stop RUN

Zero servers (firstlight_zero/server.py) register in RUN/servers, model keys are snapshot versions loaded from
RUN/weights/history/v{V}.pt. Opponent servers (firstlight_zero/il_server.py) register in RUN/il-servers. One server per listed GPU;
repeat a GPU id to run several (an opponent server's CPU work is GIL-bound, so a large engine count wants several).
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT_BASE = 31600


def step(logs, name, gpu, args):
    out = (logs / f'{name}.log').open('a')
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=str(gpu))
    return subprocess.Popen([sys.executable, *args], stdout=out, stderr=subprocess.STDOUT, env=env)


def start(run, opponent, zero_gpus, opponent_gpus):
    run = Path(run).resolve()
    if (run / 'supervisor.json').exists() and not (run / 'STOP').exists():
        raise SystemExit('training supervisor may still be running (no STOP flag); refusing to start eval servers')
    reg = run / 'servers'; il_reg = run / 'il-servers'; (run / 'eval-servers.STOP').unlink(missing_ok=True)
    for d in (reg, il_reg):
        d.mkdir(exist_ok=True)
        for old in d.glob('*.json'):
            old.unlink()
    logs = run / 'logs'; logs.mkdir(exist_ok=True); procs = []; stop_args = ['--stop-file', str(run / 'eval-servers.STOP')]
    placement = [('zero', g) for g in zero_gpus] + ([('opponent', g) for g in opponent_gpus] if opponent else [])
    for k, (kind, gpu) in enumerate(placement):
        args = ([str(ROOT / 'firstlight_zero/server.py'), '--run', str(run)] if kind == 'zero' else
                [str(ROOT / 'firstlight_zero/il_server.py'), '--checkpoint', str(Path(opponent).resolve()), '--registry', str(il_reg)])
        procs.append(step(logs, f'evalserver{k}', gpu, args + ['--port', str(PORT_BASE + k)] + stop_args))
    (run / 'eval-servers.json').write_text(json.dumps(dict(pid=os.getpid(), server_pids=[p.pid for p in procs], placement=placement, utc=time.time())))
    deadline = time.time() + 300
    count = lambda: len(list(reg.glob('*.json'))) + len(list(il_reg.glob('*.json')))
    while count() < len(placement):
        if time.time() > deadline or any(p.poll() is not None for p in procs):
            raise SystemExit(f'only {count()} of {len(placement)} servers registered; see {logs}/evalserver*.log')
        time.sleep(3)
    print(json.dumps(dict(zero=sorted(p.name for p in reg.glob('*.json')), opponent=sorted(p.name for p in il_reg.glob('*.json')))), flush=True)
    signal.signal(signal.SIGTERM, lambda *_: [p.terminate() for p in procs])
    for p in procs:
        p.wait()
    print('servers exited', [p.returncode for p in procs], flush=True)


def stop(run):
    run = Path(run).resolve(); f = run / 'eval-servers.json'
    (run / 'eval-servers.STOP').touch()
    info = json.loads(f.read_text()) if f.exists() else {}
    for pid in [info.get('pid')] + info.get('server_pids', []):
        try:
            if pid: os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    for d in ('servers', 'il-servers'):
        for old in (run / d).glob('*.json'):
            old.unlink()
    print('stopped')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('action', choices=['start', 'stop']); p.add_argument('run')
    p.add_argument('--opponent'); p.add_argument('--zero-gpus', default='0'); p.add_argument('--opponent-gpus', default='0')
    a = p.parse_args()
    if a.action == 'start':
        start(a.run, a.opponent, a.zero_gpus.split(','), a.opponent_gpus.split(','))
    else:
        stop(a.run)

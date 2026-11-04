"""Create and supervise a run: learner + inference servers on this machine, one collector per engine on every engine host.

    python firstlight_zero/launch.py prepare --run runs/NAME [--set '{"servers": 2, "server_gpus": [0, 1]}']
    python firstlight_zero/launch.py start --run runs/NAME > runs/NAME/supervisor.log 2>&1 &
    touch runs/NAME/STOP

`start` resumes from RUN/learner-state.pt when it exists. To restart in place: touch STOP, wait for the supervisor to exit,
remove STOP, start again.
"""
from pathlib import Path
import argparse, json, os, subprocess, sys, time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from firstlight_zero.hosts import ENGINES, load_engines, spawn_on   # noqa: E402

DEFAULTS = dict(
    seed=20261101, schema='firstlight-zero', engines=str(ENGINES),
    learner_gpus=[0], server_gpus=[0], servers=1,
    server_port_base=31500, server_batch_wait_ms=3.,
    gamma=.9993, lam=.95, lr=3e-4, clip=.2, value_clip=.5, epochs=2,
    batch_episodes=1024, minibatch_episodes=8, ent_coef=.003, vf_coef=.5, max_grad_norm=1.,
    max_lag=8, snapshot_every=10, self_play_fraction=.5, jitter_ticks=3,
    pfsp_power=2., pfsp_uniform=.1, pfsp_window=300,
    model=dict(channels=64, dilations=[1, 2, 4, 8], scatter_dim=16, hidden=256),
    reward=dict(),
)


def prepare(run, overrides):
    run.mkdir(parents=True, exist_ok=False)
    settings = dict(DEFAULTS, **overrides)
    (run / 'settings.json').write_text(json.dumps(settings, indent=2))
    return settings


def gpu_step(logs, name, gpus, args):
    """A local process pinned to physical GPU(s) `gpus`."""
    out = (logs / f'{name}.log').open('a')
    env = dict(os.environ, PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES=','.join(str(g) for g in gpus))
    return subprocess.Popen([sys.executable, *args], stdout=out, stderr=subprocess.STDOUT, env=env)


def learner_step(s, logs, run):
    """More than one entry in `learner_gpus` -> torchrun data-parallel learner; else one process."""
    gpus = s['learner_gpus']; script = [str(ROOT / 'firstlight_zero/learner.py'), '--run', str(run)]
    if len(gpus) > 1:   # the run path goes positional: torchrun's own parser swallows a `--run` after the script (prefix of --run-path)
        script = ['-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={len(gpus)}', str(ROOT / 'firstlight_zero/learner.py'), str(run)]
    return gpu_step(logs, 'learner', gpus, script)


def server_steps(s, logs, run):
    gpus = s['server_gpus']; procs = {}
    for k in range(int(s['servers'])):
        port = int(s['server_port_base']) + k
        procs[f'server{k}'] = gpu_step(logs, f'server{k}', [gpus[k % len(gpus)]],
                                       [str(ROOT / 'firstlight_zero/server.py'), '--run', str(run), '--port', str(port)])
    return procs


def supervise(run, collectors_per_host):
    run = run.resolve(); s = json.loads((run / 'settings.json').read_text())
    hosts = sorted({c['host'] for c in load_engines(s['engines'])['collectors']})
    logs = run / 'logs'; logs.mkdir(exist_ok=True); (run / 'servers').mkdir(exist_ok=True)
    for old in (run / 'servers').glob('*.json'): old.unlink()
    procs = {'learner': learner_step(s, logs, run)}
    deadline = time.time() + 600
    while not (run / 'weights/latest.json').exists():
        if procs['learner'].poll() is not None or time.time() > deadline:
            raise RuntimeError('learner did not publish initial weights; see logs/learner.log')
        time.sleep(2)
    procs.update(server_steps(s, logs, run))
    deadline = time.time() + 600
    while len(list((run / 'servers').glob('*.json'))) < int(s['servers']):
        if any(p.poll() is not None for p in procs.values()) or time.time() > deadline:
            raise RuntimeError('inference servers did not start; see logs/server*.log')
        time.sleep(2)
    for host in hosts:
        out = (logs / f'node-{host}.log').open('a')
        procs[host] = spawn_on(host, [sys.executable, str(ROOT / 'firstlight_zero/node.py'), '--run', str(run), '--host', host,
                                      '--limit', str(collectors_per_host)], out)
    (run / 'supervisor.json').write_text(json.dumps(dict(started=time.time(), hosts=hosts, collectors_per_host=collectors_per_host,
                                                         servers=sorted(p.name for p in (run / 'servers').glob('*.json')))))
    while not (run / 'STOP').exists():
        for name, proc in procs.items():
            if proc.poll() is not None:
                (run / 'STOP').touch()
                with (run / 'supervisor-events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(event='child_exit', name=name, code=proc.returncode, utc=time.time())) + '\n')
        time.sleep(5)
    deadline = time.time() + 180
    for proc in procs.values():
        try:
            proc.wait(max(1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.terminate()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['prepare', 'start'])
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--set', default='{}', help='JSON overrides for prepare')
    p.add_argument('--collectors-per-host', type=int, default=0, help='0 = every engine on the host')
    a = p.parse_args()
    if a.action == 'prepare':
        print(json.dumps(prepare(a.run, json.loads(a.set)), indent=2))
    else:
        supervise(a.run, a.collectors_per_host)


if __name__ == '__main__':
    main()

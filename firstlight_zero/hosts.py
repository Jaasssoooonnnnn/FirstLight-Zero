"""Where engines and workers run.

The engine list is a JSON file (default `engines.json` in the repository root, or $FIRSTLIGHT_ZERO_ENGINES):

    {"collectors": [{"host": "localhost", "port": 26789, "env_ids": [0, 1, 2, 3, 4, 5, 6, 7], "cpu": 2}, ...]}

One entry per native engine process. Engines listen on the loopback interface, so every worker runs on the engine's host:
local hosts are started directly, other hosts through ssh (the repository and the run directory must be on a shared path).
`cpu` is optional and pins the engine's worker to that core on Linux.
"""
from pathlib import Path
import json, os, shlex, socket, subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINES = Path(os.environ.get('FIRSTLIGHT_ZERO_ENGINES', ROOT / 'engines.json'))


def load_engines(path=None):
    return json.loads(Path(path or ENGINES).read_text())


def is_local(host):
    return host in ('localhost', '127.0.0.1', socket.gethostname(), socket.gethostname().split('.')[0])


def spawn_on(host, argv, stdout, env=None):
    if is_local(host):
        return subprocess.Popen(argv, stdout=stdout, stderr=subprocess.STDOUT, env=env)
    forward = {k: os.environ[k] for k in ('CR_WORKSPACE_ROOT', 'FIRSTLIGHT_ZERO_ENGINES') if k in os.environ}
    remote = ' '.join(['env', *(f'{k}={shlex.quote(v)}' for k, v in forward.items()), *map(shlex.quote, argv)])
    return subprocess.Popen(['ssh', host, remote], stdout=stdout, stderr=subprocess.STDOUT)


def pin(cpus):
    cpus = {int(c) for c in cpus if c is not None}
    if cpus and hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, cpus)

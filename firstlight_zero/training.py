"""Learner helpers: weight publishing, the inbox reader, PFSP statistics and GAE."""
from pathlib import Path
import collections, json, os, shutil, time

import numpy as np
import torch

from firstlight_zero.store import load


def publish(run, model, version, settings):
    weights = run / 'weights'; (weights / 'history').mkdir(parents=True, exist_ok=True)
    name = f'v{version:06d}.pt'
    tmp = weights / (name + '.tmp')
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, weights / name)
    tmp = weights / 'latest.json.tmp'
    tmp.write_text(json.dumps(dict(version=version, file=name, utc=time.time())))
    os.replace(tmp, weights / 'latest.json')
    if version % settings['snapshot_every'] == 0:
        shutil.copy2(weights / name, weights / 'history' / f'v{version}.pt')
    for old in sorted(weights.glob('v*.pt'))[:-4]:
        old.unlink(missing_ok=True)


def reader(run, settings, version_ref, out, stop):
    """Background thread: gathers batch_episodes fresh seat-episodes per batch, newest first. When the collectors outrun
    the learner the backlog beyond two batches is deleted unread (counted as stale) so the policy keeps training on data
    1-2 versions old instead of oldest-first consumption letting the lag grow to max_lag."""
    inbox = run / 'inbox'
    while not stop.is_set():
        episodes = []; stale = 0; broken = 0
        while len(episodes) < settings['batch_episodes'] and not stop.is_set():
            files = sorted(inbox.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
            if not files:
                time.sleep(1.); continue
            keep = 2 * settings['batch_episodes']
            for path in files[keep:]:
                path.unlink(missing_ok=True); stale += 1
            for path in files[:min(keep, len(files))][:settings['batch_episodes'] - len(episodes)]:
                try:
                    data = load(path)
                except Exception:
                    broken += 1; path.unlink(missing_ok=True); continue
                path.unlink(missing_ok=True)
                if version_ref[0] - int(data['version'].min()) > settings['max_lag']:
                    stale += 1; continue
                episodes.append(data)
        if episodes:
            out.put((episodes, stale, broken))


class Pfsp:
    """Recent score of the current policy against each history snapshot, shared with collectors via weights/pfsp.json."""

    def __init__(self, run, window):
        self.path = run / 'weights/pfsp.json'; self.window = window
        self.scores = collections.defaultdict(lambda: collections.deque(maxlen=window))
        if self.path.exists():
            for v, (n, score) in json.loads(self.path.read_text())['stats'].items():
                self.scores[int(v)].extend([score] * n)
        else:   # first start with PFSP: seed from the games the collectors already logged
            seen = []
            for f in (run / 'collectors').glob('*.jsonl'):
                for line in open(f):
                    r = json.loads(line)
                    if r.get('event') != 'game' or not r['valid']:
                        continue
                    for o, role in r['roles'].items():
                        other = r['roles'][str(1 - int(o))]
                        if role == 'current' and other != 'current':
                            seen.append((r['utc'], int(other), (r['result'][int(o)] + 1) / 2))
            for _, v, score in sorted(seen):
                self.scores[v].append(score)

    def add(self, opponent, outcome):
        self.scores[int(opponent)].append((outcome + 1) / 2)

    def write(self):
        stats = {str(v): [len(d), float(np.mean(d))] for v, d in sorted(self.scores.items()) if d}
        tmp = self.path.with_suffix('.tmp'); tmp.write_text(json.dumps(dict(utc=time.time(), window=self.window, stats=stats)))
        os.replace(tmp, self.path)


def gae(reward, value, gamma, lam):
    adv = np.zeros_like(reward); last = 0.
    for t in range(len(reward) - 1, -1, -1):
        nxt = value[t + 1] if t + 1 < len(reward) else 0.
        delta = reward[t] + gamma * nxt - value[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv

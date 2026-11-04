"""GPU PPO learner : whole-episode BPTT over seat-episodes from the inbox .

Single process:   python firstlight_zero/learner.py --run RUN
Data-parallel:    python -m torch.distributed.run --standalone --nproc_per_node=2 firstlight_zero/learner.py --run RUN
Rank 0 reads the inbox, computes advantages and the minibatch order, and broadcasts each rank its share of every
minibatch (the global minibatch stays `minibatch_episodes`); DDP averages the gradients, the loss is scaled so the
update equals the single-process one; only rank 0 publishes weights, keeps PFSP and writes metrics."""
from pathlib import Path
import argparse, collections, json, os, queue, sys, threading, time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

import numpy as np
import torch
import torch.distributed as dist

from firstlight_zero.training import Pfsp, gae, publish, reader
from firstlight_zero.model import XbowNet
from firstlight_zero.spec import ABILITY_OPTION, CARD_NAMES, SCALAR_NAMES
from firstlight_zero.store import FIXED, FLOAT_KEYS, load, unpack_units, unpack_place

ELIXIR = SCALAR_NAMES.index('my_elixir_over_10')


def elixir_stats(episodes):
    """Mean own elixir at the moment each card is played, and the share of decision frames spent sitting at 10 elixir."""
    sums = np.zeros(len(CARD_NAMES)); counts = np.zeros(len(CARD_NAMES)); capped = frames = 0
    for e in episodes:
        el = e['scal'][:, ELIXIR].astype(np.float32) * 10
        frames += len(el); capped += int((el >= 9.95).sum())
        play = (e['gate'] == 1) & (e['option'] >= 0) & (e['option'] != ABILITY_OPTION)
        if play.any():
            card = e['option_card'][play, e['option'][play]].astype(np.int64)
            np.add.at(sums, card, el[play]); np.add.at(counts, card, 1)
    return ({n: round(float(sums[i] / counts[i]), 2) for i, n in enumerate(CARD_NAMES) if counts[i]},
            float(capped / max(frames, 1)))


def collate(episodes, device):
    """Pad seat-episodes into time-major [T,B] tensors flattened to [T*B]; units padded to the minibatch maximum."""
    t_max = max(len(e['gate']) for e in episodes); b = len(episodes)
    width = max(1, max(int(e['unit_n'].max()) if len(e['unit_n']) else 0 for e in episodes))

    def pad(arrays, dtype):
        first = arrays[0]
        out = np.zeros((t_max, b) + first.shape[1:], dtype)
        for j, arr in enumerate(arrays):
            out[:len(arr), j] = arr
        return torch.from_numpy(out).to(device, non_blocking=True).reshape((t_max * b,) + first.shape[1:])
    x = {k: pad([e[k] for e in episodes], np.float32 if k in FLOAT_KEYS else np.int64) for k in FIXED}
    units = [unpack_units(e, width) for e in episodes]
    x['unit_type'] = pad([ut for ut, _ in units], np.int64); x['unit_feat'] = pad([uf for _, uf in units], np.float32)
    act = dict(gate=pad([e['gate'] for e in episodes], np.int64), option=pad([e['option'] for e in episodes], np.int64),
               cell=pad([e['cell'] for e in episodes], np.int64), gate_free=pad([e['gate_free'] for e in episodes], bool),
               opt_mask=pad([e['opt_mask'] for e in episodes], bool), place=pad([unpack_place(e) for e in episodes], bool))
    extra = {k: pad([e[k] for e in episodes], np.float32) for k in ('logp', 'adv', 'ret', 'value')}
    valid = pad([np.ones(len(e['gate']), bool) for e in episodes], bool)
    return x, act, extra, valid, t_max


class Evaluate(torch.nn.Module):
    """DDP synchronises gradients only for calls that go through forward(), so PPO evaluation is routed through it."""

    def __init__(self, net):
        super().__init__(); self.net = net

    def forward(self, x, act, time_steps):
        return self.net.evaluate(x, act, time_steps)


def wrap(net, world):
    model = Evaluate(net)
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=[world.local]) if world.world > 1 else model


class Dist:
    """World layout: torchrun sets RANK/WORLD_SIZE/LOCAL_RANK; without them this is a single process."""

    def __init__(self):
        self.world = int(os.environ.get('WORLD_SIZE', 1)); self.rank = int(os.environ.get('RANK', 0))
        self.local = int(os.environ.get('LOCAL_RANK', 0)); self.device = torch.device('cuda', self.local)
        torch.cuda.set_device(self.device)
        if self.world > 1:
            dist.init_process_group('nccl', device_id=self.device)

    def broadcast(self, obj):
        if self.world == 1:
            return obj
        box = [obj]; dist.broadcast_object_list(box, src=0, device=self.device); return box[0]

    def total(self, value):
        """Sum of a scalar over ranks."""
        if self.world == 1:
            return float(value)
        t = torch.tensor([float(value)], device=self.device); dist.all_reduce(t); return float(t)


def plan_batch(episodes, s, rng):
    """Rank 0: advantages, normalisation and the minibatch order for one update. Returns per-step rank shares."""
    for e in episodes:
        e['adv'] = gae(e['reward'], e['value'], s['gamma'], s['lam']).astype(np.float32); e['ret'] = e['adv'] + e['value']
    adv_all = np.concatenate([e['adv'] for e in episodes]); mean, std = float(adv_all.mean()), float(adv_all.std() + 1e-8)
    order = sorted(range(len(episodes)), key=lambda j: len(episodes[j]['gate']))
    chunks = [order[i:i + s['minibatch_episodes']] for i in range(0, len(order), s['minibatch_episodes'])]
    steps = [chunks[k] for epoch in range(s['epochs']) for k in rng.permutation(len(chunks))]
    return dict(mean=mean, std=std, steps=steps)


def update(model, opt, episodes, plan, s, world):
    """One PPO update over `plan['steps']` (lists of episode indices); this rank handles its slice of every step.
    Returns per-step stats as (weighted sums, frame count) pairs that are summed over ranks by the caller."""
    rank = world.rank; stats = collections.defaultdict(float); frames = 0.
    model.train()
    for members in plan['steps']:
        mine = members[rank::world.world]                       # rank r takes episodes r, r+W, ... of the minibatch
        n_total = world.total(sum(len(episodes[j]['gate']) for j in mine))
        if not mine:                                            # fewer episodes than ranks: still take part in the all-reduce
            loss = sum(p.sum() for p in model.parameters()) * 0.
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); continue
        x, act, ex, valid, t_max = collate([episodes[j] for j in mine], world.device)
        logp, entropy, value = model(x, act, t_max)
        adv = (ex['adv'] - plan['mean']) / plan['std']; ratio = torch.exp(logp - ex['logp'])
        pg = -torch.min(ratio * adv, ratio.clamp(1 - s['clip'], 1 + s['clip']) * adv)
        v_clip = ex['value'] + (value - ex['value']).clamp(-s['value_clip'], s['value_clip'])
        vl = torch.max((value - ex['ret']) ** 2, (v_clip - ex['ret']) ** 2)
        w = valid.float()
        # DDP averages gradients over ranks: scale by world so the sum over all ranks' frames / n_total is recovered
        loss = ((pg + s['vf_coef'] * .5 * vl - s['ent_coef'] * entropy) * w).sum() * world.world / n_total
        opt.zero_grad(set_to_none=True); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), s['max_grad_norm']); opt.step()
        with torch.no_grad():
            lr_ = logp - ex['logp']; n = float(w.sum())
            stats['approx_kl'] += float(((torch.exp(lr_) - 1 - lr_) * w).sum())
            stats['clip_frac'] += float((((ratio - 1).abs() > s['clip']).float() * w).sum())
            stats['entropy'] += float((entropy * w).sum()); stats['value_loss'] += float((vl * w).sum())
            stats['grad_norm'] += float(grad) * n; frames += n
    return stats, frames


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', type=Path); p.add_argument('run_positional', nargs='?', type=Path)
    a = p.parse_args(); run = a.run or a.run_positional     # torchrun swallows a `--run` option of its script (matches --run-path)
    s = json.loads((run / 'settings.json').read_text())
    world = Dist(); device = world.device; main_rank = world.rank == 0
    torch.manual_seed(s['seed'] + world.rank)
    torch.backends.cudnn.benchmark = False   # shapes change every minibatch; benchmarking would re-tune each step
    net = XbowNet(**s['model']).to(device)
    if main_rank:
        print('parameters', sum(p.numel() for p in net.parameters()), 'ranks', world.world, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=s['lr'], eps=1e-5)
    version = 0
    resume = run / 'learner-state.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        net.load_state_dict(state['model']); opt.load_state_dict(state['opt']); version = state['version']
    model = wrap(net, world)
    rng = np.random.default_rng(s['seed'] + version)
    if main_rank:
        publish(run, net, version, s)
        version_ref = [version]; out = queue.Queue(maxsize=2); stop = threading.Event()
        threading.Thread(target=reader, args=(run, s, version_ref, out, stop), daemon=True).start()
        metrics = run / 'metrics.jsonl'; started = time.time(); games_seen = 0
        pfsp = Pfsp(run, s.get('pfsp_window', 300)); pfsp.write()
    try:
        while True:
            payload = None
            if main_rank:
                while payload is None and not (run / 'STOP').exists():
                    try:
                        episodes, stale, broken = out.get(timeout=5)
                        payload = dict(episodes=episodes, stale=stale, broken=broken, plan=plan_batch(episodes, s, rng))
                    except queue.Empty:
                        continue
            payload = world.broadcast(payload)          # None = stop (rank 0 saw STOP)
            if payload is None:
                break
            episodes, plan = payload['episodes'], payload['plan']
            t0 = time.time()
            stats, frames = update(model, opt, episodes, plan, s, world)
            frames_all = world.total(frames); stats = {k: world.total(v) / max(frames_all, 1.) for k, v in stats.items()}
            version += 1
            if not main_rank:
                continue
            version_ref[0] = version
            publish(run, net, version, s)
            if version % 10 == 0:
                torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), version=version), run / 'learner-state.pt.tmp')
                os.replace(run / 'learner-state.pt.tmp', run / 'learner-state.pt')
            values = np.concatenate([e['value'] for e in episodes]); rets = np.concatenate([e['ret'] for e in episodes])
            by_age = collections.defaultdict(lambda: [0, 0.]); unique = set()
            for e in episodes:
                m = e['meta']; unique.add(m['game'])
                if m['opponent'] != 'current':
                    age = int(e['version'].max()) - int(m['opponent'])
                    bucket = 'age<=20' if age <= 20 else 'age<=60' if age <= 60 else 'age>60'
                    by_age[bucket][0] += 1; by_age[bucket][1] += (m['outcome'] + 1) / 2; pfsp.add(m['opponent'], m['outcome'])
            pfsp.write(); games_seen += len(unique)
            metas = [e['meta'] for e in episodes if 'plays_by_card' in e['meta']]
            plays = (dict(zip(CARD_NAMES, np.mean([m['plays_by_card'] for m in metas], 0).round(2).tolist())) if metas else {})
            elixir_at_play, frames_at_10 = elixir_stats(episodes)
            record = dict(version=version, utc=time.time(), episodes=len(episodes), stale=payload['stale'], broken=payload['broken'],
                          gamma=s['gamma'], epochs=s['epochs'], ranks=world.world,
                          plays_per_game=plays, elixir_at_play=elixir_at_play, frames_at_10=frames_at_10,
                          game_seconds=float(np.mean([m['tick'] for m in metas]) / 20) if metas else None,
                          acts_per_game=float(np.mean([m['acts'] for m in metas])) if metas else None,
                          crowns_per_game=float(np.mean([m['crowns'] for m in metas])) if metas else None,
                          decisions=int(sum(len(e['gate']) for e in episodes)), mean_len=float(np.mean([len(e['gate']) for e in episodes])),
                          act_rate=float(np.mean(np.concatenate([e['gate'] for e in episodes]))),
                          mean_reward=float(np.mean([e['reward'].sum() for e in episodes])),
                          lag_mean=float(np.mean([version - 1 - e['version'].mean() for e in episodes])),
                          explained_var=float(1 - np.var(rets - values) / (np.var(rets) + 1e-8)),
                          vs_history={k: dict(games=v[0], score=v[1] / max(v[0], 1)) for k, v in by_age.items()},
                          update_s=time.time() - t0, games_per_hour=games_seen / max(time.time() - started, 1) * 3600, **stats)
            with metrics.open('a') as f:
                f.write(json.dumps(record) + '\n')
    finally:
        if main_rank:
            stop.set()
            torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), version=version), run / 'learner-state.pt')
        if world.world > 1:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

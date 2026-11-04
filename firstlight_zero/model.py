"""Pruned IL policy for the X-Bow mirror: units scattered onto the board -> dilated residual CNN (full-field
receptive field) -> LSTM(256) -> gate / option pointer / FiLM-conditioned location / value.

Per 250 ms decision: WAIT or ACT; one option (4 hand slots or the Hero Knight ability); a cell for cards.
Delay is the earliest legal bin (no delay head); one card per decision (no continue head).
"""
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .spec import (ABILITY_OPTION, CARD_FEATURES, CELLS, HEIGHT, NO_CARD, OPP_CARD_FEATURES, OPTIONS, OPTION_FEATURES,
                   SCALARS, UNIT_FEATURES, UNIT_KIND, UNIT_VOCAB, WIDTH)

NEG = -1e9
EXPLICIT_PLANES = 8     # own/enemy x (characters, buildings, projectiles, hp/5000)
POS_PLANES = 8          # learned board geometry


def mlp(*sizes):
    layers = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers[:-1])


class ResidualBlock(nn.Module):
    """The original ResidualSpatialBlock with a dilation."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels); self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(8, channels); self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        nn.init.zeros_(self.conv2.weight); nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        r = self.conv1(F.silu(self.norm1(x)))
        return x + self.conv2(F.silu(self.norm2(r)))


class XbowNet(nn.Module):
    def __init__(self, channels=96, dilations=(1, 2, 4, 8), scatter_dim=16, hidden=256, card_dim=32, key_dim=32):
        super().__init__()
        self.channels, self.hidden = channels, hidden
        self.unit_type = nn.Embedding(UNIT_VOCAB, 16)
        self.unit = mlp(16 + UNIT_FEATURES, 64, scatter_dim)
        self.register_buffer('unit_kind', torch.tensor(UNIT_KIND, dtype=torch.long), persistent=False)
        self.pos = nn.Parameter(torch.zeros(POS_PLANES, HEIGHT, WIDTH))
        ys, xs = torch.meshgrid(torch.arange(HEIGHT), torch.arange(WIDTH), indexing='ij')
        self.register_buffer('coords', torch.stack([(xs + .5) / WIDTH, (ys + .5) / HEIGHT]).float(), persistent=False)
        in_ch = scatter_dim + EXPLICIT_PLANES + POS_PLANES + 2
        self.inp = nn.Conv2d(in_ch, channels, 3, padding=1)
        self.blocks = nn.Sequential(*(ResidualBlock(channels, d) for d in dilations))
        self.inp.to(memory_format=torch.channels_last); self.blocks.to(memory_format=torch.channels_last)
        self.card = nn.Embedding(NO_CARD + 1, card_dim)
        self.vec = mlp(SCALARS + 8 * CARD_FEATURES + 8 * OPP_CARD_FEATURES + OPTIONS * (card_dim + OPTION_FEATURES), 256, 128)
        self.trunk = nn.Sequential(nn.Linear(2 * channels + 128, hidden), nn.ReLU())
        self.lstm = nn.LSTM(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden, 2)
        self.option_query = nn.Linear(hidden, 64)
        self.option_key = mlp(card_dim + OPTION_FEATURES + 1, 64, 64)
        self.cond = mlp(hidden + card_dim + 1, 128, 128)
        self.film = nn.Linear(128, 2 * channels)
        self.loc_key = nn.Conv2d(channels, key_dim, 1)
        self.loc_query = nn.Linear(128, key_dim)
        self.value = mlp(hidden + 2 * channels, 128, 1)
        with torch.no_grad():   # start rarely acting (P(ACT)=0.15) with near-uniform cells
            self.gate.bias.copy_(torch.tensor([math.log(.85), math.log(.15)])); self.gate.weight.mul_(.01)
            self.film.weight.mul_(.01); self.film.bias.zero_()
            self.loc_query.weight.mul_(.01); self.loc_query.bias.zero_()

    def initial_state(self, batch, device=None):
        zeros = torch.zeros(1, batch, self.hidden, device=device)
        return zeros, zeros.clone()

    # ------------------------------------------------------------------ board
    def board(self, x):
        """Scatter units bilinearly onto the grid and add the explicit / positional planes. -> [N, C_in, H, W]"""
        unit_type, feat = x['unit_type'], x['unit_feat']
        n, u = unit_type.shape
        valid = unit_type > 0
        v = self.unit(torch.cat([self.unit_type(unit_type), feat], -1)).float()         # [N,U,S]; scatter accumulates in fp32 even under autocast
        v = v * valid.unsqueeze(-1)
        px = (feat[..., 1] * WIDTH - .5).clamp(0, WIDTH - 1 - 1e-4); py = (feat[..., 2] * HEIGHT - .5).clamp(0, HEIGHT - 1 - 1e-4)
        x0, y0 = px.floor(), py.floor(); fx, fy = px - x0, py - y0
        x0, y0 = x0.long(), y0.long(); x1 = (x0 + 1).clamp(max=WIDTH - 1); y1 = (y0 + 1).clamp(max=HEIGHT - 1)
        cells = torch.stack([y0 * WIDTH + x0, y0 * WIDTH + x1, y1 * WIDTH + x0, y1 * WIDTH + x1], -1)        # [N,U,4]
        w = torch.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], -1) * valid.unsqueeze(-1)
        base = (torch.arange(n, device=feat.device) * CELLS).view(n, 1, 1)
        flat = (cells + base).reshape(-1)
        scat = torch.zeros(n * CELLS, v.shape[-1], device=feat.device, dtype=v.dtype)
        scat.index_add_(0, flat, (v.unsqueeze(2) * w.unsqueeze(-1)).reshape(-1, v.shape[-1]))
        scat = scat.view(n, HEIGHT, WIDTH, -1).permute(0, 3, 1, 2)
        # explicit planes: nearest cell counts by kind and hp, per side
        kind = self.unit_kind[unit_type]                                                   # [N,U]
        mine = feat[..., 0] > .5
        side = torch.where(mine, 0, 4)
        plane = torch.where(kind == 3, 1, kind) + side                                      # towers count as buildings
        nearest = (py.round().long().clamp(0, HEIGHT - 1) * WIDTH + px.round().long().clamp(0, WIDTH - 1) + base.view(n, 1))
        explicit = torch.zeros(n, EXPLICIT_PLANES * CELLS, device=feat.device, dtype=feat.dtype)
        idx = (plane * CELLS + nearest - base.view(n, 1))
        explicit.scatter_add_(1, idx, valid.to(feat.dtype))
        explicit.scatter_add_(1, (side + 3) * CELLS + nearest - base.view(n, 1), feat[..., 4] * valid)
        explicit = explicit.view(n, EXPLICIT_PLANES, HEIGHT, WIDTH)
        return torch.cat([scat, explicit, self.pos.unsqueeze(0).expand(n, -1, -1, -1), self.coords.unsqueeze(0).expand(n, -1, -1, -1)], 1)

    def grid(self, x, chunk_rows=12288):
        """CNN feature map [N,C,H,W] (bf16 on CUDA). Under grad, minibatches above chunk_rows rows are checkpointed in
        chunks so the peak holds one chunk's activations (about 3.4 MB per row at 64 channels). A learner minibatch of
        8 episodes (~9.4k rows) is not checkpointed: recomputing the CNN cost 28% of the step (657 -> 470 ms,
        tools/ckpt_check.py) for a peak of 32 GiB instead of 6 on an 80 GB card."""
        planes = self.board(x).contiguous(memory_format=torch.channels_last)
        if not torch.is_grad_enabled() or planes.shape[0] <= chunk_rows:
            return self._cnn(planes)
        return torch.cat([checkpoint(self._cnn, planes[i:i + chunk_rows], use_reentrant=False)
                          for i in range(0, planes.shape[0], chunk_rows)])

    def _cnn(self, planes):
        with torch.autocast(device_type=planes.device.type, dtype=torch.bfloat16, enabled=planes.device.type == 'cuda'):
            return self.blocks(self.inp(planes))

    # ------------------------------------------------------------------ core
    def encode(self, x):
        g = self.grid(x)                                                                    # [N,C,H,W]
        summary = torch.cat([g.float().mean((2, 3)), g.float().amax((2, 3))], -1)          # [N,2C]
        n = x['scal'].shape[0]
        vec = torch.cat([x['scal'], x['cards'].reshape(n, -1), x['opp_cards'].reshape(n, -1),
                         self.card(x['option_card']).reshape(n, -1), x['opt_feat'].reshape(n, -1)], -1)
        return self.trunk(torch.cat([summary, self.vec(vec)], -1)), g, summary

    def core(self, x, state, time_steps):
        z, g, summary = self.encode(x)
        t = time_steps; b = z.shape[0] // t
        out, state = self.lstm(z.view(t, b, -1), state)
        return self.norm(out.reshape(t * b, -1) + z), state, g, summary

    # ------------------------------------------------------------------ heads
    def option_logits(self, h, x, opt_mask):
        is_ability = torch.zeros_like(x['opt_feat'][..., :1]); is_ability[:, ABILITY_OPTION] = 1.
        key = self.option_key(torch.cat([self.card(x['option_card']), x['opt_feat'], is_ability], -1))
        logits = (self.option_query(h).unsqueeze(1) * key).sum(-1) / 8.
        return logits.masked_fill(~opt_mask, NEG)

    def location_logits(self, h, x, g, option, place):
        card = self.card(x['option_card'].gather(1, option.clamp_min(0).unsqueeze(1)).squeeze(1))
        cond = self.cond(torch.cat([h, card, (option == ABILITY_OPTION).float().unsqueeze(1)], -1))
        gamma, beta = self.film(cond).chunk(2, -1)
        keys = self.loc_key(g.float() * (1 + gamma[:, :, None, None]) + beta[:, :, None, None])     # [N,K,H,W]
        logits = torch.einsum('nk,nkhw->nhw', self.loc_query(cond), keys).reshape(g.shape[0], -1) / math.sqrt(keys.shape[1])
        return logits.masked_fill(~place.reshape(place.shape[0], -1), NEG)

    # ------------------------------------------------------------------ acting
    @torch.no_grad()
    def act(self, x, legality, state, greedy=False, temperature=1.0):
        """Eval-time acting. `temperature` divides the logits of every head (gate, option, location); greedy = argmax.
        Training never calls this (the inference server samples from heads() at T=1)."""
        if not greedy and temperature <= 0:
            greedy = True
        tau = 1.0 if greedy else float(temperature)
        h, state, g, summary = self.core(x, state, 1)
        n = h.shape[0]; dev = h.device
        opt_mask = legality['opt_mask']; any_legal = opt_mask.any(1)
        force = legality['force'] & any_legal; gate_free = any_legal & ~force
        gate_logits = self.gate(h) / tau; gate_dist = torch.distributions.Categorical(logits=gate_logits)
        gate = gate_logits.argmax(-1) if greedy else gate_dist.sample()
        gate = torch.where(force, torch.ones_like(gate), torch.where(any_legal, gate, torch.zeros_like(gate)))
        logp = torch.where(gate_free, gate_dist.log_prob(gate), torch.zeros(n, device=dev))
        option = torch.full((n,), -1, dtype=torch.long, device=dev); cell = torch.full((n,), -1, dtype=torch.long, device=dev)
        idx = (gate == 1).nonzero().squeeze(1)
        if idx.numel():
            sub = {k: v[idx] for k, v in x.items()}
            o_logits = self.option_logits(h[idx], sub, opt_mask[idx]) / tau; o_dist = torch.distributions.Categorical(logits=o_logits)
            o = o_logits.argmax(-1) if greedy else o_dist.sample()
            option[idx] = o; logp[idx] += o_dist.log_prob(o)
            card_rows = (o != ABILITY_OPTION).nonzero().squeeze(1)
            if card_rows.numel():
                rows = idx[card_rows]
                place = legality['place'][rows, o[card_rows]]
                l_logits = self.location_logits(h[rows], {k: v[rows] for k, v in x.items()}, g[rows], o[card_rows], place) / tau
                l_dist = torch.distributions.Categorical(logits=l_logits)
                c = l_logits.argmax(-1) if greedy else l_dist.sample()
                cell[rows] = c; logp[rows] += l_dist.log_prob(c)
        value = self.value(torch.cat([h, summary], -1)).squeeze(-1)
        return dict(gate=gate, option=option, cell=cell, gate_free=gate_free, logp=logp, value=value), state

    # ------------------------------------------------------------------ graph-friendly acting (inference server)
    def heads(self, x, legality, state):
        """One decision for every row with static shapes: logits of all heads (location for all four card options),
        value and the new recurrent state. No sampling, no data-dependent indexing, so it can be CUDA-graph captured."""
        h, state, g, summary = self.core(x, state, 1)
        n = h.shape[0]
        opt_mask = legality['opt_mask']
        option_logits = self.option_logits(h, x, opt_mask | ~opt_mask.any(1, keepdim=True))   # rows w/o legal options: flat
        cards = self.card(x['option_card'][:, :4])                                                    # [N,4,D]
        cond = self.cond(torch.cat([h.unsqueeze(1).expand(-1, 4, -1), cards, torch.zeros(n, 4, 1, device=h.device)], -1))
        gamma, beta = self.film(cond).chunk(2, -1)                                                    # [N,4,C]
        conditioned = (g.float().unsqueeze(1) * (1 + gamma[..., None, None]) + beta[..., None, None]).reshape(n * 4, *g.shape[1:])
        keys = self.loc_key(conditioned).reshape(n, 4, -1, g.shape[2], g.shape[3])                     # [N,4,K,H,W]
        loc = torch.einsum('nok,nokhw->nohw', self.loc_query(cond), keys).reshape(n, 4, -1) / math.sqrt(keys.shape[2])
        loc = loc.masked_fill(~legality['place'].reshape(n, 4, -1), NEG)
        value = self.value(torch.cat([h, summary], -1)).squeeze(-1)
        return dict(gate=self.gate(h), option=option_logits, location=loc, value=value), state

    @torch.no_grad()
    def sample(self, heads, legality, greedy=False):
        """Turn `heads` into actions with the same distributions as act()."""
        gate_logits, option_logits, loc, value = heads['gate'], heads['option'], heads['location'], heads['value']
        n = gate_logits.shape[0]; dev = gate_logits.device
        opt_mask = legality['opt_mask']; any_legal = opt_mask.any(1)
        force = legality['force'] & any_legal; gate_free = any_legal & ~force
        gate_dist = torch.distributions.Categorical(logits=gate_logits)
        gate = gate_logits.argmax(-1) if greedy else gate_dist.sample()
        gate = torch.where(force, torch.ones_like(gate), torch.where(any_legal, gate, torch.zeros_like(gate)))
        logp = torch.where(gate_free, gate_dist.log_prob(gate), torch.zeros(n, device=dev))
        o_dist = torch.distributions.Categorical(logits=option_logits)
        o = option_logits.argmax(-1) if greedy else o_dist.sample()
        acting = gate == 1
        option = torch.where(acting, o, torch.full_like(o, -1))
        logp = logp + torch.where(acting, o_dist.log_prob(o), torch.zeros_like(logp))
        is_card = acting & (o != ABILITY_OPTION)
        sel = loc.gather(1, o.clamp(0, 3).view(n, 1, 1).expand(-1, 1, loc.shape[-1])).squeeze(1)
        sel = torch.where(is_card.unsqueeze(1), sel, torch.zeros_like(sel))
        l_dist = torch.distributions.Categorical(logits=sel)
        c = sel.argmax(-1) if greedy else l_dist.sample()
        cell = torch.where(is_card, c, torch.full_like(c, -1))
        logp = logp + torch.where(is_card, l_dist.log_prob(c), torch.zeros_like(logp))
        return dict(gate=gate, option=option, cell=cell, gate_free=gate_free, logp=logp, value=value)

    def sample_t(self, heads, legality, temp):
        """Eval-time sample() with a per-row temperature `temp` [N] dividing the gate, option and location logits; rows with
        temp <= 0 take the argmax. Same structure as sample(); training only ever uses sample()."""
        gate_logits, option_logits, loc, value = heads['gate'], heads['option'], heads['location'], heads['value']
        n = gate_logits.shape[0]; dev = gate_logits.device
        greedy = temp <= 0; tau = torch.where(greedy, torch.ones_like(temp), temp).unsqueeze(1)
        opt_mask = legality['opt_mask']; any_legal = opt_mask.any(1)
        force = legality['force'] & any_legal
        pick = lambda logits: torch.where(greedy, logits.argmax(-1), torch.distributions.Categorical(logits=logits / tau).sample())
        gate = pick(gate_logits)
        gate = torch.where(force, torch.ones_like(gate), torch.where(any_legal, gate, torch.zeros_like(gate)))
        o = pick(option_logits)
        acting = gate == 1
        option = torch.where(acting, o, torch.full_like(o, -1))
        is_card = acting & (o != ABILITY_OPTION)
        sel = loc.gather(1, o.clamp(0, 3).view(n, 1, 1).expand(-1, 1, loc.shape[-1])).squeeze(1)
        sel = torch.where(is_card.unsqueeze(1), sel, torch.zeros_like(sel))
        c = pick(sel)
        cell = torch.where(is_card, c, torch.full_like(c, -1))
        zero = torch.zeros(n, device=dev)
        return dict(gate=gate, option=option, cell=cell, gate_free=any_legal & ~force, logp=zero, value=value)

    # ------------------------------------------------------------------ learning
    def evaluate(self, x, act, time_steps):
        """Teacher-forced log-prob, entropy and value over [T,B] episodes from a zero state."""
        n = x['scal'].shape[0]; dev = x['scal'].device
        h, _, g, summary = self.core(x, self.initial_state(n // time_steps, dev), time_steps)
        gate_lp = F.log_softmax(self.gate(h), -1); free = act['gate_free']
        logp = torch.where(free, gate_lp.gather(1, act['gate'].unsqueeze(1)).squeeze(1), torch.zeros(n, device=dev))
        entropy = torch.where(free, -(gate_lp.exp() * gate_lp).sum(-1), torch.zeros(n, device=dev))
        acting = act['gate'] == 1; option = act['option'].clamp_min(0)
        o_lp = F.log_softmax(self.option_logits(h, x, act['opt_mask'] | ~acting.unsqueeze(1)), -1)
        o_ent = -(o_lp.exp() * o_lp.clamp_min(-30.)).sum(-1)
        logp = logp + torch.where(acting, o_lp.gather(1, option.unsqueeze(1)).squeeze(1), torch.zeros_like(logp))
        entropy = entropy + torch.where(acting, o_ent, torch.zeros_like(entropy))
        rows = (acting & (act['option'] != ABILITY_OPTION)).nonzero().squeeze(1)   # the location head only for card plays
        if rows.numel():
            l_lp = F.log_softmax(self.location_logits(h[rows], {k: v[rows] for k, v in x.items()}, g[rows], option[rows], act['place'][rows]), -1)
            l_ent = -(l_lp.exp() * l_lp.clamp_min(-30.)).sum(-1)
            logp = logp.index_add(0, rows, l_lp.gather(1, act['cell'][rows].clamp_min(0).unsqueeze(1)).squeeze(1))
            entropy = entropy.index_add(0, rows, l_ent)
        return logp, entropy, self.value(torch.cat([h, summary], -1)).squeeze(-1)

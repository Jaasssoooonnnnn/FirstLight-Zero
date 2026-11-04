"""CUDA-graph replay of XbowNet.heads for one (model, batch-size bucket): static buffers, one launch per batch."""
import torch

from .spec import HEIGHT, MAX_UNITS, OPTIONS, OPTION_FEATURES, SCALARS, CARD_FEATURES, OPP_CARD_FEATURES, UNIT_FEATURES, WIDTH

BUCKETS = (16, 32, 64, 128, 256, 512, 1024)
COARSE_BUCKETS = (16, 64, 256, 1024)   # history snapshots; 16 keeps a graph per rarely drawn snapshot small
LONG = ('unit_type', 'option_card')


def bucket(n, coarse=False):
    for b in (COARSE_BUCKETS if coarse else BUCKETS):
        if n <= b:
            return b
    raise ValueError(f'batch of {n} seats exceeds the largest bucket')


class GraphRunner:
    def __init__(self, model, batch, device):
        self.model, self.b, self.device = model, batch, device
        z = lambda *shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype, device=device)
        self.x = dict(unit_type=z(batch, MAX_UNITS, dtype=torch.long), unit_feat=z(batch, MAX_UNITS, UNIT_FEATURES), scal=z(batch, SCALARS),
                      cards=z(batch, 8, CARD_FEATURES), opp_cards=z(batch, 8, OPP_CARD_FEATURES),
                      option_card=z(batch, OPTIONS, dtype=torch.long), opt_feat=z(batch, OPTIONS, OPTION_FEATURES))
        self.legal = dict(opt_mask=z(batch, OPTIONS, dtype=torch.bool), place=z(batch, 4, HEIGHT, WIDTH, dtype=torch.bool))
        self.state = (z(1, batch, model.hidden), z(1, batch, model.hidden))
        stream = torch.cuda.Stream(device); stream.wait_stream(torch.cuda.current_stream(device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for _ in range(3):
                model.heads(self.x, self.legal, self.state)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, capture_error_mode='thread_local'):
            self.out, self.new_state = model.heads(self.x, self.legal, self.state)

    def __call__(self, x, legal, state):
        """Inputs: tensors with n <= bucket rows (already on the device). Returns heads and new state for the first n rows."""
        n = x['scal'].shape[0]
        for k, v in self.x.items():
            v[:n].copy_(x[k]); 
            if n < self.b: v[n:].zero_()
        for k, v in self.legal.items():
            v[:n].copy_(legal[k])
            if n < self.b: v[n:].zero_()
        for dst, src in zip(self.state, state):
            dst[:, :n].copy_(src)
        self.graph.replay()
        return {k: v[:n] for k, v in self.out.items()}, (self.new_state[0][:, :n], self.new_state[1][:, :n])

from dataclasses import dataclass

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv

from einops import einsum, rearrange, repeat

from layers.ssm import InterpolationTokenMixer, Conv1DTokenMixer


@dataclass
class InitStrategyS6(object):
    """As SSMs are potentially sensible to initialization, wrap the init strategy"""
    A: str = 'hippo'  # {hippo, random, constant}
    # A: str = 'random'
    delta: str = None  # Bias only

    def _init_A(self, log_nA):
        if self.A == 'hippo':
            size = log_nA.size()
            n_layers, d_input, d_state = size
            log_nA.fill_(0).add_(
                repeat(
                    torch.log(torch.arange(1, d_state + 1) + 1),
                    'd -> n b d',
                    n=n_layers,
                    b=d_input
                )
            )
        elif self.A == 'random':
            nn.init.xavier_uniform_(log_nA)
        else:
            nn.init.constant_(log_nA, np.log(0.5))

    def _init_delta(self, delta):
        delta.fill_(0).add_(
            torch.tensor(np.log(np.exp(np.random.uniform(0.001, 0.1, delta.size())) - 1))
        )

    @torch.no_grad()
    def init(self, log_nA, delta):
        self._init_A(log_nA)
        self._init_delta(delta)


class DiagonalS6SSM(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        ssm_format: str = 'siso',  # {siso, mimo}
        d_state: int = 16,
        num_layers: int = 2,
        token_mixer: str = 'conv1d',  # {conv1d, interp, None}
        pre_token_mix: bool = False,
        bn: bool = False,
        layer: str = 'sage',        
    ):
        super(DiagonalS6SSM, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.token_mixers = nn.ModuleList()
        # channel_mixers are only present in siso formats
        self.channel_mixers = nn.ModuleList()
        self.residual_connections = nn.ModuleList()
        # dimensions
        self.d_model = hidden_channels
        self.d_state = d_state

        # params
        self.log_nA = nn.Parameter(torch.empty(self.num_layers, self.d_model, self.d_state))
        self.delta = nn.Parameter(torch.empty(self.num_layers, self.d_model))

        self.pre_token_mixers = nn.ModuleList()
        self.pre_token_mix = pre_token_mix

        self.pre_token_mixers.append(
            Conv1DTokenMixer(in_channels)
            if token_mixer == 'conv1d' else InterpolationTokenMixer(in_channels)
        )

        for i in range(num_layers):
            first_channel = in_channels if i == 0 else hidden_channels
            self.bns.append(nn.BatchNorm1d(first_channel) if bn else nn.Identity())
            self.convs.append(
                SAGEConv(first_channel, hidden_channels * 2 + d_state * 2)
                if layer == 'sage' else GCNConv(first_channel, hidden_channels * 2 + d_state * 2)
            )
            self.token_mixers.append(
                Conv1DTokenMixer(hidden_channels)
                if token_mixer == 'conv1d' else InterpolationTokenMixer(hidden_channels)
            )
            self.residual_connections.append(
                nn.Linear(first_channel, hidden_channels)
            )
            self.channel_mixers.append(
                nn.Linear(d_state * hidden_channels, hidden_channels)
            )

        self.mlp = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()

        self.init_strategy = InitStrategyS6()
        self.reset_parameters()

    def reset_parameters(self):
        self.init_strategy.init(self.log_nA, self.delta)

    def forward(self, snapshots):
        xs = [data.x for data in snapshots]
        # if self.pre_token_mix:
        #     xs = self.pre_token_mixers[0](xs)
        #     xs = [x.squeeze() for x in xs.chunk(xs.size(0))]
        v = xs[0].size(0)

        for i in range(self.num_layers):
            xsr = [self.residual_connections[i](x) for x in xs]
            dts, Bs, Cs = [], [], []
            # Step 1: Do the graph convolutions
            for j, data in enumerate(snapshots):
                # xs[j] = self.convs[i](xs[j], data.edge_index)
                outs = self.convs[i](xs[j], data.edge_index)
                xs[j], dt, B, C = torch.split(
                    outs,
                    [self.d_model, self.d_model, self.d_state, self.d_state],
                    dim=-1
                )
                dts.append(dt)
                Bs.append(B)
                Cs.append(C)

            # Step 2: Token mixing
            xs_ = self.token_mixers[i](xs) if i == 0 else xs
            # xs_ = xs
            # Step 3: State computation
            state = None
            for j in range(len(xs)):
                # Step 2: state update
                A = - torch.exp(self.log_nA[i])
                dt = F.softplus(dts[j] + self.delta[i].unsqueeze(0))
                A_zoh = torch.exp(einsum(dt, A, 'v d, d n -> v d n'))
                B = einsum(dt, Bs[j], 'v d, v n -> v d n')
                C = Cs[j]

                x = xs_[j]
                v, d = x.size()
                if state is None:
                    state = x.new_zeros(v, d, self.d_state)
                B_x = einsum(B, x, 'v d n, v d -> v d n')
                state = A_zoh * state + B_x
                xs[j] = self.activation(einsum(state, C, 'v d n, v n -> v d'))
                xs[j]= F.layer_norm(xs[j] + xsr[j], (d,))
                # xs[j] = xs[j] + xsr[j]
        return self.mlp(xs[-1])

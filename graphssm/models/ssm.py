import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv

from layers.ssm import DiagonalSISOCell, DiagonalMIMOCell, InterpolationTokenMixer, Conv1DTokenMixer


class DiagonalSSM(nn.Module):
    """"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        ssm_format: str = 'siso',  # {siso, mimo}
        hidden_channels: int = 64,
        d_state: int = 16,
        num_layers: int = 2,
        token_mixer: str = 'conv1d',  # {conv1d, interp, None}
        pre_token_mix: bool = True,
        bn: bool = False,
        layer: str = 'sage',
    ):
        super(DiagonalSSM, self).__init__()
        self.num_layers = num_layers
        self.format = ssm_format
        self.ssm_cells = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.token_mixers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # channel_mixers are only present in siso formats
        self.channel_mixers = nn.ModuleList()
        self.residual_connections = nn.ModuleList()

        self.pre_token_mixers = nn.ModuleList()
        self.pre_token_mix = pre_token_mix

        self.pre_token_mixers.append(
            Conv1DTokenMixer(in_channels)
            if token_mixer == 'conv1d' else InterpolationTokenMixer(in_channels)
        )
        
        for i in range(num_layers):
            first_channel = in_channels if i == 0 else hidden_channels
            self.bns.append(nn.BatchNorm1d(first_channel) if bn else nn.Identity())
            self.convs.append(SAGEConv(first_channel, hidden_channels) if layer == 'sage' else GCNConv(first_channel, hidden_channels))
            self.token_mixers.append(
                Conv1DTokenMixer(hidden_channels)
                if token_mixer == 'conv1d' else InterpolationTokenMixer(hidden_channels)
            )
            self.residual_connections.append(
                nn.Linear(first_channel, hidden_channels)
            )
            if self.format == 'siso':
                self.channel_mixers.append(
                    nn.Linear(d_state * hidden_channels, hidden_channels)
                )
                self.ssm_cells.append(
                    DiagonalSISOCell(
                        d_state=d_state,
                        d_input=hidden_channels
                    )
                )
            else:
                self.ssm_cells.append(
                    DiagonalMIMOCell(
                        d_state=hidden_channels,
                        d_input=hidden_channels
                    )
                )

        self.mlp = nn.Linear(hidden_channels, out_channels)
        self.activation = nn.ReLU()

    def forward(self, snapshots):
        xs = [data.x for data in snapshots]
        if self.pre_token_mix:
            xs = self.pre_token_mixers[0](xs)
            xs = [x.squeeze() for x in xs.chunk(xs.size(0))]
        v = xs[0].size(0)

        for i in range(self.num_layers):
            xsr = [self.residual_connections[i](x) for x in xs]
            # Step 1: Do the graph convolutions
            for j, data in enumerate(snapshots):
                xs[j] = self.bns[i](xs[j])
                xs[j] = self.convs[i](xs[j], data.edge_index)

            # Step 2: Token mixing
            if not self.pre_token_mix:
                # 只对0层做token mixer
                if i == 0:
                    xs_ = self.token_mixers[i](xs)

            # Step 3: State computation
            state = None
            for j in range(len(xs)):
                # Step 2: state update
                if self.pre_token_mix:
                    state = self.ssm_cells[i](xs[j], state)
                else:
                    state = self.ssm_cells[i](xs_[j], state)
                # Step 3: channel mixing
                if self.format == 'siso':
                    xs[j] = self.channel_mixers[i](
                        self.activation(state.reshape(v, -1)))
                else:
                    xs[j] = self.activation(state)
                xs[j] = xs[j] + xsr[j]
        return self.mlp(xs[-1])
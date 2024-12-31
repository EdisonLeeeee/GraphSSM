from dataclasses import dataclass

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat


@dataclass
class InitStrategy(object):
    """As SSMs are potentially sensible to initialization, wrap the init strategy"""
    A: str = 'hippo'  # {hippo, random, constant}
    B: str = 'constant'  # {hippo, random, constant}
    C: str = None  # doesn't matter that much
    delta: str = None  # TODO: explore initialization schemes here

    def _init_A(self, log_nA):
        if self.A == 'hippo':
            size = log_nA.size()
            if len(size) == 2:
                d_input, d_state = size
                log_nA.fill_(0).add_(
                    # torch.tile(
                    #     torch.log(torch.arange(1, self.d_state + 1) + 1).view(1, -1),
                    #     [self.d_input, 1]
                    # )
                    repeat(
                        torch.log(torch.arange(1, d_state + 1) + 1),
                        'd -> b d',
                        b=d_input
                    )
                )
            else:
                d_state = size[0]
                log_nA.fill_(0).add_(torch.log(torch.arange(1, d_state + 1) + 1))
        elif self.A == 'random':
            nn.init.xavier_uniform_(log_nA)
        else:
            nn.init.constant_(log_nA, np.log(0.5))

    def _init_B(self, B):
        if self.B == 'hippo':
            d_input, d_state = B.size()
            B.fill_(0).add_(
                repeat(
                    torch.sqrt(2 * torch.arange(1, d_state + 1) + 1),
                    'd -> b d',
                    b=d_input
                )
            )
        elif self.B == 'constant':
            nn.init.constant_(B, 1.)
        else:
            nn.init.xavier_uniform_(B)

    def _init_C(self, C):
        nn.init.xavier_uniform_(C)

    def _init_delta(self, delta: nn.Linear):
        nn.init.constant_(delta.weight, 0.)
        nn.init.constant_(
            delta.bias,
            np.log(np.exp(np.random.uniform(0.001, 0.1)) - 1)
        )

    @torch.no_grad()
    def init(self, log_nA, B, C, delta):
        self._init_A(log_nA)
        self._init_B(B)
        self._init_C(C)
        self._init_delta(delta)


class Conv1DTokenMixer(nn.Module):
    """Token mixing using Conv1d, similar to Mamba"""

    def __init__(self, d_input, window_size: int = 2, use_padding=True):
        super(Conv1DTokenMixer, self).__init__()
        self.use_padding = use_padding
        self.window_size = window_size
        self.conv = nn.Conv1d(
            in_channels=d_input,
            out_channels=d_input,
            kernel_size=window_size,
            padding='same' if use_padding else 'valid'
        )

    def forward(self, xs):
        xs = torch.stack(xs, dim=-1)
        x_conv = self.conv(xs)
        if not self.use_padding:
            # TODO: keeping the raw appears problematic
            x_conv = torch.cat(
                [torch.stack(xs[:(self.window_size - 1)], dim=-1), x_conv],
                dim=-1
            )
        return rearrange(x_conv, 'v d l->l v d')


class InterpolationTokenMixer(nn.Module):
    """Token mixing using the prescribed approach in the note
    TODO: implement the Hyena-style parameterized convolution?
    """

    def __init__(self, d_input):
        super(InterpolationTokenMixer, self).__init__()
        self.interp = nn.Sequential(
            nn.Linear(2 * d_input, d_input),
            nn.Sigmoid()
        )
        self.stretch = nn.Sequential(
            nn.Linear(2 * d_input, d_input),
            nn.Softplus()
        )

    def forward(self, xs):
        out = []
        last_x = None
        for x in xs:
            if last_x is None:
                out.append(x)
            else:
                x_ = torch.cat([last_x, x], dim=1)
                interp_weight = self.interp(x_)  # [V, D]
                scale = self.stretch(x_)  # [V, D]
                out.append(
                    scale * (interp_weight * last_x + (1 - interp_weight) * x)
                )
            last_x = x
        return torch.stack(out, dim=0)  # type consistency with conv1d mixer


class DiagonalSISOCell(nn.Module):
    """SISO impl of state-space layers that is linear time invariant and diagonal
    The impl is based on real parameters (instead of complex ones)

    Conceptually, think of this as an S4D-Real cell with optional handling of adaptive step size

    TODO:
        - maybe enable scan-based impl later (for performance improvement)
        - enable handling of time encodings based on given time gaps
    """

    def __init__(self, d_state, d_input, init_strategy: InitStrategy = None, **kwargs):
        super(DiagonalSISOCell, self).__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.init_strategy = init_strategy or InitStrategy()
        # For state update, we allow input-dependent adaptive time gaps here as
        # this is not a convolutional impl
        self.log_nA = nn.Parameter(torch.empty(d_input, d_state))  # log(-A)
        self.B = nn.Parameter(torch.empty(d_input, d_state))
        self.C = nn.Parameter(torch.empty(d_input, d_state, d_state))
        self.s_delta = nn.Sequential(
            # Here the bias term might require specific initialization, see mamba paper
            nn.Linear(d_input, 1),
            nn.Softplus()
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.init_strategy.init(self.log_nA, self.B, self.C, self.s_delta[0])

    def forward(self, x, state=None, delta=None):
        """State update

        Args:
            x (torch.Tensor): input tensor of shape [V, D]
            state (torch.Tensor): current state of shape [V, D, N]
            delta (torch.Tensor): time gap, either a scalar or node-dependent vector

        Returns:
            updated state of shape [V, D, N], no mixing operations are performed
        """
        v, d = x.size()
        if state is None:
            state = x.new_zeros(v, d, self.d_state)
        if delta is None:
            delta = self.s_delta(x).squeeze(1)  # [V]
        else:  # Delta is a scalar, no check here
            delta = torch.ones(v, device=x.device) * delta
        A = - torch.exp(self.log_nA)
        A_zoh = torch.exp(einsum(delta, A, 'v, d n -> v d n'))
        B = einsum(delta, self.B, 'v, d n -> v d n')  # Using the exp(x) ~ 1 + x approximation
        B_x = einsum(B, x, 'v d n, v d -> v d n')
        state = A_zoh * state + B_x
        return einsum(state, self.C, 'v d n, d n q -> v d q')


class DiagonalMIMOCell(nn.Module):
    """MIMO impl conceptually think of it as an S5 cell with optional adaptive step size
    """

    def __init__(self, d_state, d_input, init_strategy: InitStrategy = None, **kwargs):  # tying d_input and d_output
        super(DiagonalMIMOCell, self).__init__()
        self.d_state = d_state
        self.d_input = d_input
        self.init_strategy = init_strategy or InitStrategy()
        self.log_nA = nn.Parameter(torch.empty(d_state))  # log(-A) in the diagonal form
        self.B = nn.Parameter(torch.empty(d_input, d_state))
        self.s_delta = nn.Sequential(
            # Here the bias term might require specific initialization, see mamba paper
            nn.Linear(d_input, 1),
            nn.Softplus()
        )
        self.C = nn.Parameter(torch.empty(d_state, d_input))
        self.reset_parameters()

    def reset_parameters(self):
        self.init_strategy.init(self.log_nA, self.B, self.C, self.s_delta[0])

    def forward(self, x, state=None, delta=None):
        v, d = x.size()
        if state is None:
            state = torch.zeros(v, self.d_state, device=x.device)
        if delta is None:
            delta = self.s_delta(x).squeeze(1)  # [V]
        else:  # Delta is a scalar, no check here
            delta = torch.ones(v, device=x.device) * delta
        A = - torch.exp(self.log_nA)
        A_zoh = torch.exp(einsum(delta, A, 'v, d -> v d'))
        B_x = einsum(delta, self.B, x, 'v, d n, v d -> v n')
        # B_x = einsum(B, x, 'v d n, v d -> v n')
        state = A_zoh * state + B_x
        return state @ self.C

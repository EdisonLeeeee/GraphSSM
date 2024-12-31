from typing import Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage
from torch_geometric.transforms import BaseTransform

from graphssm.data import TemporalData

class TemporalSplit(BaseTransform):
    def __init__(
        self,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        key: Optional[str] = "t",
    ):
        # TODO (jintang): accept integer inputs
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.key = key

    def forward(self, data: TemporalData) -> TemporalData:
        key = self.key
        t = data[key].sort().values
        val_ratio = self.val_ratio
        test_ratio = self.test_ratio
        val_time, test_time = np.quantile(
            t.cpu().numpy(), [1. - val_ratio - test_ratio, 1. - test_ratio])
        data.train_mask = data[key] < val_time
        data.val_mask = torch.logical_and(data[key] >= val_time, data[key]
                                          < test_time)
        data.test_mask = data[key] >= test_time
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.val_ratio}, '
                f'{self.test_ratio})')


class ToTemporalUndirected(BaseTransform):
    def forward(self, data: TemporalData) -> TemporalData:
        src, dst = data.src, data.dst
        num_events = src.size(0)
        edge_dir = torch.cat([torch.zeros_like(src), torch.ones_like(dst)])

        # TODO: considering self-loops
        src, dst = torch.cat([src, dst]), torch.cat([dst, src])
        for key, value in data._store.items():
            if key in ['src', 'dst', 'edge_label_index', 'edge_label']:
                continue
            if not isinstance(value, torch.Tensor) or value.dim() == 0:
                continue
            if value.size(0) == num_events:
                if value.dtype == torch.bool:
                    # we don't make bool masks undirected here
                    data[key] = torch.cat([value, torch.zeros_like(value)])
                else:
                    data[key] = torch.cat([value, value])
        data.src, data.dst = src, dst
        data.edge_dir = edge_dir
        return data



class RandomNodeSplit(BaseTransform):
    def __init__(
        self,
        num_splits: int = 1,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
        unknown: Optional[int] = None,
    ):
        self.num_splits = num_splits
        self.num_val = num_val
        self.num_test = num_test
        self.key = key
        self.unknown = unknown

    def forward(
        self,
        data: Union[Data, HeteroData, TemporalData],
    ) -> Union[Data, HeteroData, TemporalData]:
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_samples = num_nodes = store.num_nodes
        perm = torch.randperm(num_nodes)
        if self.unknown is not None:
            y = getattr(store, self.key)
            nodes = torch.where(y != self.unknown)[0]
            perm = nodes[torch.randperm(nodes.size(0))]
            num_samples = perm.size(0)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if isinstance(self.num_val, float):
            num_val = round(num_samples * self.num_val)
        else:
            num_val = self.num_val

        if isinstance(self.num_test, float):
            num_test = round(num_samples * self.num_test)
        else:
            num_test = self.num_test

        val_mask[perm[:num_val]] = True
        test_mask[perm[num_val:num_val + num_test]] = True
        train_mask[perm[num_val + num_test:]] = True

        return train_mask, val_mask, test_mask


class StratifyNodeSplit(BaseTransform):
    def __init__(
        self,
        num_splits: int = 1,
        num_val: Union[int, float] = 500,
        num_test: Union[int, float] = 1000,
        key: Optional[str] = "y",
        unknown: Optional[int] = None,
    ):
        self.num_splits = num_splits
        self.num_val = num_val
        self.num_test = num_test
        self.key = key
        self.unknown = unknown

    def forward(
        self,
        data: Union[Data, HeteroData, TemporalData],
    ) -> Union[Data, HeteroData, TemporalData]:
        for store in data.node_stores:
            if self.key is not None and not hasattr(store, self.key):
                continue

            train_masks, val_masks, test_masks = zip(
                *[self._split(store) for _ in range(self.num_splits)])

            store.train_mask = torch.stack(train_masks, dim=-1).squeeze(-1)
            store.val_mask = torch.stack(val_masks, dim=-1).squeeze(-1)
            store.test_mask = torch.stack(test_masks, dim=-1).squeeze(-1)

        return data

    def _split(self, store: NodeStorage) -> Tuple[Tensor, Tensor, Tensor]:
        num_nodes = store.num_nodes
        perm = torch.randperm(num_nodes)
        y = getattr(store, self.key)
        if self.unknown is not None:
            nodes = torch.where(y != self.unknown)[0]
            perm = nodes[torch.randperm(nodes.size(0))]

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        from sklearn.model_selection import train_test_split
        train_perm, test_perm = train_test_split(perm, test_size=self.num_test,
                                                 stratify=y[perm])

        train_perm, val_perm = train_test_split(
            train_perm,
            test_size=self.num_val / (1 - self.num_val - self.num_val),
            stratify=y[train_perm])

        val_mask[val_perm] = True
        test_mask[test_perm] = True
        train_mask[train_perm] = True

        return train_mask, val_mask, test_mask
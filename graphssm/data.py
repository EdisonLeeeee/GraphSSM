import copy
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.data import BaseData, size_repr
from torch_geometric.data.storage import (
    E_KEYS,
    N_KEYS,
    BaseStorage,
    EdgeStorage,
    GlobalStorage,
    NodeStorage,
)
from torch_geometric.utils import subgraph

E_KEYS = E_KEYS | {'src', 'dst', 't', 'msg', 'edge_dir'}


class TemporalData(BaseData):
    r"""A data object composed by a stream of events describing a temporal
    graph.
    The :class:`~torch_geometric.data.TemporalData` object can hold a list of
    events (that can be understood as temporal edges in a graph) with
    structured messages.
    An event is composed by a source node, a destination node, a timestamp
    and a message. Any *Continuous-Time Dynamic Graph* (CTDG) can be
    represented with these four values.

    In general, :class:`~torch_geometric.data.TemporalData` tries to mimic
    the behavior of a regular Python dictionary.
    In addition, it provides useful functionality for analyzing graph
    structures, and provides basic PyTorch tensor functionalities.

    .. code-block:: python

        from torch import Tensor
        from torch_geometric.data import TemporalData

        events = TemporalData(
            src=Tensor([1,2,3,4]),
            dst=Tensor([2,3,4,5]),
            t=Tensor([1000,1010,1100,2000]),
            msg=Tensor([1,1,0,0])
        )

        # Add additional arguments to `events`:
        events.y = Tensor([1,1,0,0])

        # It is also possible to set additional arguments in the constructor
        events = TemporalData(
            ...,
            y=Tensor([1,1,0,0])
        )

        # Get the number of events:
        events.num_events
        >>> 4

        # Analyzing the graph structure:
        events.num_nodes
        >>> 5

        # PyTorch tensor functionality:
        events = events.pin_memory()
        events = events.to('cuda:0', non_blocking=True)

    Parameters
    ----------
    src : Optional[Tensor], optional
        A list of source nodes for the events
        with shape :obj:`[num_events]`, by default :obj:`None`
    dst : Optional[Tensor], optional
        A list of destination nodes for the
        events with shape :obj:`[num_events]`, by default :obj:`None`
    t : Optional[Tensor], optional
        The timestamps for each event with shape
        :obj:`[num_events]`, by default :obj:`None`
    msg : Optional[Tensor], optional
        Messages feature matrix with shape
        :obj:`[num_events, num_msg_features]`, by default :obj:`None`

    .. note::
        The shape of :obj:`src`, :obj:`dst`, :obj:`t`
        and the first dimension of :obj`msg` should be
        the same (:obj:`num_events`).
    """
    def __init__(
        self,
        src: Optional[Tensor] = None,
        dst: Optional[Tensor] = None,
        t: Optional[Tensor] = None,
        msg: Optional[Tensor] = None,
        **kwargs,
    ):
        super().__init__()
        self.__dict__['_store'] = GlobalStorage(_parent=self)

        self.src = src
        self.dst = dst
        self.t = t
        self.msg = msg

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, mapping: Dict[str, Any]) -> 'TemporalData':
        r"""Creates a :class:`~torch_geometric.data.TemporalData` object from
        a Python dictionary."""
        return cls(**mapping)

    def merge(
        self,
        *,
        step: Optional[int] = None,
        unit: Optional[Union[int, float]] = None,
    ) -> 'TemporalData':
        assert step is None or unit is None
        data = copy.copy(self)
        if step is not None:
            # merged by a consecutive time step
            assert step > 1, step
            t = data.t.clone()
            iterator = data.time_stamps[::step]
            for i in range(iterator.size(0) - 1):
                start = iterator[i]
                end = iterator[i + 1]
                mask = torch.logical_and(t >= start, t < end)
                t[mask] = start
            t[t >= iterator[-1]] = iterator[-1]
            data.t = t
        elif unit is not None:
            # divided by a time unit
            data.t = (data.t / unit).to(torch.long)
        return data

    def drop_duplicates(self) -> 'TemporalData':
        data = copy.copy(self)
        unique, idx, counts = torch.unique(self.edge_index, dim=1, sorted=True,
                                           return_inverse=True,
                                           return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0]), cum_sum[:-1]))
        first_indicies = ind_sorted[cum_sum]
        return data[first_indicies]

    def snapshots(self) -> List['TemporalData']:
        return [self.snapshot(start=i) for i in range(self.num_snapshots)]

    def snapshot(
        self,
        start: int = 0,
        end: Optional[int] = None,
        relabel_nodes: bool = True,
        last_node_attr: bool = False,
    ) -> 'TemporalData':
        """Returns the graph snapshot where events start from
        :obj:`start`-th and end at :obj:`end`-th time step.

        Parameters
        ----------
        start: Optional[int]
            the starting index of the snapshot, if :obj:`None`,
            :obj:`0` is used, by default :obj:`None`
        end: Optional[int]
            the ending index of the snapshot, if :obj:`None`,
            :obj:`-1` is used, by default :obj:`None`

        Examples:
        ---------
        .. code-block:: python

            from torch import Tensor
            from lastgl.data import TemporalData

            events = TemporalData(
                src=Tensor([1,2,3,4]),
                dst=Tensor([2,3,4,5]),
                t=Tensor([1000,1010,1100,2000]),
                msg=Tensor([1,1,0,0])
            )
            >>> events.snapshot().edge_index
            tensor([[1., 2., 3., 4.],
                    [2., 3., 4., 5.]])

            >>> events.snapshot(0,2).edge_index
            tensor([[1., 2., 3.],
                    [2., 3., 4.]])
        """
        t = self.t
        time_stamps = self.time_stamps
        end = end or start
        t_start = time_stamps[start]

        if end == start:
            mask = t == t_start
        elif end >= time_stamps.size(0):
            mask = t >= t_start
        else:
            t_end = time_stamps[end]
            mask = torch.logical_and(t >= t_start, t <= t_end)

        data = copy.copy(self)
        num_events = data.num_events
        num_nodes = data.num_nodes
        if t.size(0) == num_events:
            for key, value in self._store.items():
                if not isinstance(value, Tensor) or value.dim() == 0:
                    continue
                if value.size(0) == num_events:
                    data[key] = value[mask]
                elif value.size(0) == num_nodes:
                    if value.dim() == 3:
                        # [num_nodes, num_timesteps, num_features]
                        if end is None:
                            value = value[:, start:, :]
                        else:
                            value = value[:, start:end + 1, :]

                        if last_node_attr:
                            value = value[:, -1, :]
                        data[key] = value.squeeze(1)

        elif t.size(0) == num_nodes:
            edge_index, _, edge_mask = subgraph(mask, self.edge_index,
                                                relabel_nodes=relabel_nodes,
                                                return_edge_mask=True)
            for key, value in self._store.items():
                if key in ['src', 'dst']:
                    continue
                if not isinstance(value, Tensor) or value.dim() == 0:
                    continue
                if value.size(0) == num_events:
                    data[key] = value[edge_mask]
                elif value.size(0) == num_nodes and relabel_nodes:
                    value = value[mask]
                    data[key] = value
                    data.num_nodes = value.size(0)
            data.src, data.dst = edge_index[0], edge_index[1]
        else:
            raise ValueError
        return data

    def is_node_attr(self, key, val):
        num_nodes = self.num_nodes
        if key in N_KEYS:
            return True
        if key in E_KEYS:
            return False
        if not isinstance(val, Tensor) or val.dim() == 0:
            return False
        if val.size(0) == num_nodes:
            return True
        return False

    def is_edge_attr(self, key, val):
        num_events = self.num_events
        if key in E_KEYS:
            return True
        if key in N_KEYS:
            return False
        if not isinstance(val, Tensor) or val.dim() == 0:
            return False
        if val.size(0) == num_events:
            return True
        return False

    def index_select(self, idx: Any) -> 'TemporalData':
        idx = prepare_idx(idx)
        data = copy.copy(self)
        for key, value in data._store.items():
            if not isinstance(value, Tensor) or value.dim() == 0:
                continue
            if value.size(0) == self.num_events:
                data[key] = value[idx]
        return data

    def __getitem__(self, idx: Any) -> Any:
        if isinstance(idx, str):
            return self._store[idx]
        return self.index_select(idx)

    def __setitem__(self, key: str, value: Any):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        self._store[key] = value

    def __delitem__(self, key: str):
        if key in self._store:
            del self._store[key]

    def __getattr__(self, key: str) -> Any:
        if '_store' not in self.__dict__:
            raise RuntimeError(
                "The 'data' object was created by an older version of PyG. "
                "If this error occurred while loading an already existing "
                "dataset, remove the 'processed/' directory in the dataset's "
                "root folder and try again.")
        return getattr(self._store, key)

    def __setattr__(self, key: str, value: Any):
        setattr(self._store, key, value)

    def __delattr__(self, key: str):
        delattr(self._store, key)

    def __iter__(self) -> Iterable:
        for i in range(self.num_events):
            yield self[i]

    def __len__(self) -> int:
        return self.num_events

    def __call__(self, *args: List[str]) -> Iterable:
        for key, value in self._store.items(*args):
            yield key, value

    def __copy__(self):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = value
        out.__dict__['_store'] = copy.copy(self._store)
        out._store._parent = out
        return out

    def __deepcopy__(self, memo):
        out = self.__class__.__new__(self.__class__)
        for key, value in self.__dict__.items():
            out.__dict__[key] = copy.deepcopy(value, memo)
        out._store._parent = out
        return out

    def stores_as(self, data: 'TemporalData'):
        return self

    @property
    def stores(self) -> List[BaseStorage]:
        return [self._store]

    @property
    def node_stores(self) -> List[NodeStorage]:
        return [self._store]

    @property
    def edge_stores(self) -> List[EdgeStorage]:
        return [self._store]

    def to_dict(self) -> Dict[str, Any]:
        return self._store.to_dict()

    def to_namedtuple(self) -> NamedTuple:
        return self._store.to_namedtuple()

    def to_static(self) -> Data:
        edge_index = self.edge_index
        edge_attr = self.get('msg')
        edge_time = self.get('t')
        static_dict = self.to_dict()
        static_dict.pop('src', None)
        static_dict.pop('dst', None)
        static_dict.pop('msg', None)
        static_dict.pop('t', None)
        static_dict['edge_index'] = edge_index
        static_dict['edge_attr'] = edge_attr
        static_dict['edge_time'] = edge_time
        return Data(**static_dict)

    def debug(self):
        pass  # TODO

    @property
    def num_nodes(self) -> Optional[int]:
        r"""Returns the number of nodes in the graph."""
        # We sequentially access attributes that reveal the number of nodes.
        store = self._store
        if 'num_nodes' in store:
            return store['num_nodes']
        for key, value in store.items():
            if isinstance(value, Tensor) and key in N_KEYS:
                cat_dim = store._parent().__cat_dim__(key, value, store)
                return value.size(cat_dim)
        return max(int(self.src.max()), int(self.dst.max())) + 1

    @property
    def num_events(self) -> int:
        r"""Returns the number of events loaded.

        .. note::
            In a :class:`~torch_geometric.data.TemporalData`, each row denotes
            an event.
            Thus, they can be also understood as edges.
        """
        if self.src is None:
            return 0
        return self.src.size(0)

    @property
    def num_edges(self) -> int:
        r"""Alias for :meth:`~torch_geometric.data.TemporalData.num_events`."""
        return self.num_events

    @property
    def num_snapshots(self) -> int:
        r"""Returns the number of snapshots loaded."""
        # TODO: use cached attributes to aviod torch.unique
        return self.time_stamps.size(0)

    @property
    def time_stamps(self) -> Tensor:
        return torch.unique(self.t, sorted=True)  # TODO: use LRU cache

    @property
    def edge_index(self) -> Tensor:
        r"""Returns the edge indices of the graph."""
        if 'edge_index' in self:
            return self._store['edge_index']
        if self.src is not None and self.dst is not None:
            return torch.stack([self.src, self.dst], dim=0)
        raise ValueError(f"{self.__class__.__name__} does not contain "
                         f"'edge_index' information")

    def size(
        self, dim: Optional[int] = None
    ) -> Union[Tuple[Optional[int], Optional[int]], Optional[int]]:
        r"""Returns the size of the adjacency matrix induced by the graph."""
        size = (int(self.src.max()), int(self.dst.max()))
        return size if dim is None else size[dim]

    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if 'batch' in key:
            return int(value.max()) + 1
        elif key in ['src', 'dst']:
            return self.num_nodes
        else:
            return 0

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        info = ', '.join([size_repr(k, v) for k, v in self._store.items()])
        return f'{cls}({info})'

    def transpose(self) -> 'TemporalData':
        data = copy.copy(self)
        data.src, data.dst = data.dst, data.src
        return data

    def triplets(self) -> Tuple[Tensor, Tensor, Tensor]:
        return self.src, self.dst, self.t

    def coalesce(self):
        raise NotImplementedError

    def has_isolated_nodes(self) -> bool:
        raise NotImplementedError

    def has_self_loops(self) -> bool:
        raise NotImplementedError

    def is_undirected(self) -> bool:
        raise NotImplementedError

    def is_directed(self) -> bool:
        raise NotImplementedError


###############################################################################


def prepare_idx(idx):
    if isinstance(idx, int):
        return slice(idx, idx + 1)
    if isinstance(idx, (list, tuple)):
        return torch.tensor(idx)
    elif isinstance(idx, slice):
        return idx
    elif isinstance(idx, torch.Tensor) and idx.dtype == torch.long:
        return idx
    elif isinstance(idx, torch.Tensor) and idx.dtype == torch.bool:
        return idx

    raise IndexError(
        f"Only strings, integers, slices (`:`), list, tuples, and long or "
        f"bool tensors are valid indices (got '{type(idx).__name__}')")

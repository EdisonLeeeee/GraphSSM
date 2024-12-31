import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, download_url

from graphssm.data import TemporalData


class DBLP(InMemoryDataset):
    url = ("https://www.dropbox.com/sh/palzyh5box1uc1v/"
           "AACSLHB7PChT-ruN-rksZTCYa?dl=0")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return ['dblp.txt', 'node2label.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        src = []
        dst = []
        t = []
        path = os.path.join(self.raw_dir, 'dblp.txt')
        with open(path) as f:
            for line in f:
                x, y, z = line.strip().split()
                src.append(int(x))
                dst.append(int(y))
                t.append(float(z))
        num_nodes = max(max(src), max(dst)) + 1
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        t = torch.tensor(t, dtype=torch.float)

        t, perm = t.sort()
        src = src[perm]
        dst = dst[perm]

        nodes = []
        labels = []
        path = os.path.join(self.raw_dir, 'node2label.txt')
        with open(path) as f:
            for line in f:
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(int(label))

        from sklearn.preprocessing import LabelEncoder
        labels = LabelEncoder().fit_transform(labels)
        y = torch.zeros(num_nodes, dtype=torch.long)
        y[nodes] = torch.tensor(labels, dtype=torch.long)

        path = os.path.join(self.raw_dir, 'dblp.npy')
        if os.path.exists(path):
            print('Loading processed node features...')
            x = np.load(path)
            x = torch.tensor(x).to(torch.float).transpose(0, 1).contiguous()
        else:
            x = None

        data = TemporalData(src=src, dst=dst, t=t, y=y, x=x,
                            num_nodes=num_nodes)
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])

class Tmall(InMemoryDataset):
    url = ("https://www.dropbox.com/sh/palzyh5box1uc1v/"
           "AACSLHB7PChT-ruN-rksZTCYa?dl=0")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return ['tmall.txt', 'node2label.txt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        src = []
        dst = []
        t = []
        path = os.path.join(self.raw_dir, 'tmall.txt')
        with open(path) as f:
            for line in f:
                x, y, z = line.strip().split()
                src.append(int(x))
                dst.append(int(y))
                t.append(float(z))
        num_nodes = max(max(src), max(dst)) + 1
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)
        t = torch.tensor(t, dtype=torch.float)

        t, perm = t.sort()
        src = src[perm]
        dst = dst[perm]

        nodes = []
        labels = []
        path = os.path.join(self.raw_dir, 'node2label.txt')
        with open(path) as f:
            for line in f:
                node, label = line.strip().split()
                nodes.append(int(node))
                labels.append(int(label))

        from sklearn.preprocessing import LabelEncoder
        labels = LabelEncoder().fit_transform(labels)
        y = torch.full((num_nodes, ), -1, dtype=torch.long)
        y[nodes] = torch.tensor(labels, dtype=torch.long)

        data = TemporalData(src=src, dst=dst, t=t, y=y, num_nodes=num_nodes)

        path = os.path.join(self.raw_dir, 'tmall.npy')
        if os.path.exists(path):
            print('Loading processed node features...')
            x = np.load(path)
            x = torch.tensor(x).to(torch.float).transpose(0, 1).contiguous()
            # reindexing
            # according to the SpikeNet paper
            others = set(range(num_nodes)) - set(nodes)
            all_nodes = nodes + list(others)
            new_x = torch.zeros_like(x)
            new_x[all_nodes] = x
            x = new_x
            # Merge snapshots with a window size 10,
            # according to the SpikeNet paper
            data = data.merge(step=10)
        else:
            x = None
        data.x = x
        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])



class STARDataset(InMemoryDataset):
    url = 'https://github.com/EdisonLeeeee/TemporalDatasets/raw/master/{}.npz'

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ):
        self.name = name.lower()
        assert self.name in ['dblp3', 'dblp5', 'reddit', 'brain']

        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> str:
        return f'{self.name}.npz'

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        download_url(self.url.format(self.name), self.raw_dir)

    def process(self):
        file = np.load(self.raw_paths[0])
        
        x = file['attmats']  # (N, T, D)
        y = file['labels']  # (N, C)
        adjs = file['adjs']  # (T, N, N)

        x = torch.from_numpy(x).to(torch.float)
        y = torch.from_numpy(y.argmax(1)).to(torch.long)
        t = []
        src = []
        dst = []
        for i, adj in enumerate(adjs):
            row, col = adj.nonzero()
            src.append(torch.from_numpy(row).to(torch.long))
            dst.append(torch.from_numpy(col).to(torch.long))
            t.append(torch.full((src[-1].size(0), ), i, dtype=torch.long))
        t = torch.cat(t)
        src = torch.cat(src)
        dst = torch.cat(dst)
        data = TemporalData(src=src, dst=dst, t=t, x=x, y=y)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'
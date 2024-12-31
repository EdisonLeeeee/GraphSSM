import torch
import argparse
import time
import torch.nn.functional as F
from copy import copy
from tqdm import tqdm
from loguru import logger
from sklearn import metrics
from torch_geometric.transforms import AddSelfLoops, Compose, ToUndirected
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from torch_geometric import seed_everything
from torch_geometric.utils import subgraph
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset

from graphssm.transforms import RandomNodeSplit, ToTemporalUndirected, StratifyNodeSplit
from graphssm.datasets import DBLP, STARDataset, Tmall

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--dataset', type=str, default="arxiv")
parser.add_argument('--hidden_channels', type=int, default=64)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--ssm_format', type=str, default='siso')
parser.add_argument('--token_mixer', type=str, default='interp')
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--val_size', type=float, default=0.05)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_name', type=str, default="ssm")
args = parser.parse_args()

seed_everything(args.seed)

args = parser.parse_args()

config = {
    'hidden_channels': args.hidden_channels,
    'learning_rate': args.learning_rate,
    "weight_decay": args.weight_decay,
    "ssm_format": args.ssm_format,
    "token_mixer": args.token_mixer,
    'epochs': args.epochs,
    "seed": args.seed
}


device = f'cuda:{args.device}'

##################################################


root = './data'
dataset = 'ogbn-arxiv'
dataset = PygNodePropPredDataset(
root=root, name=dataset, transform=T.ToUndirected()
)

data = dataset[0]
data.y = data.y.squeeze()
data.node_year = data.node_year.squeeze()
edge_time = torch.zeros(data.edge_index.size(1))
for year in data.node_year.unique():
    mask = data.node_year == year
    edge_index, _, edge_mask = subgraph(mask, edge_index=data.edge_index, num_nodes=data.num_nodes, return_edge_mask=True)
    edge_time[edge_mask] = year
    
data.edge_time = edge_time
    
bins = []
snapshots = []
for year in data.node_year.unique():
    mask = data.node_year <= year
    edge_index = subgraph(mask, edge_index=data.edge_index, num_nodes=data.num_nodes)[0]
    if edge_index.size(1) <= 1000: continue
    bins.append(year)
    snapshot = copy(data)
    snapshot.edge_index = edge_index
    snapshots.append(snapshot)

train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
split_idx = dataset.get_idx_split()
train_mask[split_idx["train"]] = True
val_mask[split_idx["valid"]] = True
test_mask[split_idx["test"]] = True
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
##################################################
num_nodes = data.num_nodes


if args.model_name == "ssm":
    from models.ssm import DiagonalSSM
elif args.model_name == "s6":
    from models.s6 import DiagonalS6SSM as DiagonalSSM

model = DiagonalSSM(
    data.x.size(-1),
    data.y.max().item()+1,
    hidden_channels=config["hidden_channels"],
    ssm_format=config["ssm_format"],
    token_mixer=config["token_mixer"],
    bn=True,
    layer='gcn',
).to(device)

print(f'#Paras {sum(p.numel()/1e3 for p in model.parameters()):.3f}K')
optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
train_loader = NeighborLoader(data,
                              # num_neighbors=[-1, -1],
                              num_neighbors=[5, 5],
                              batch_size=args.batch_size,
                              shuffle=True,
                              input_nodes=data.train_mask)


def get_subgraph_snapshots(batch):
    edge_index = batch.edge_index
    edge_time = batch.edge_time
    subgraphs = []
    for i, t in enumerate(bins[:-1]):
        mask = edge_time <= t
        g = Data(x=batch.x, edge_index=edge_index[:, mask])
        subgraphs.append(g)
    batch.edge_index = edge_index
    subgraphs.append(batch)
    return subgraphs


def train(snapshots):
    model.train()
    total_loss = 0.
    for batch in tqdm(train_loader):
        subgraphs = get_subgraph_snapshots(batch)
        subgraphs = [subgraph.to(device) for subgraph in subgraphs]
        batch = subgraphs[-1]
        batch_size = batch.batch_size
        optimizer.zero_grad()
        out = model(subgraphs)
        loss = F.cross_entropy(out[:batch_size], batch.y[:batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss


@torch.no_grad()
def test(snapshots, mask):
    test_loader = NeighborLoader(data,
                                  num_neighbors=[-1, -1],
                                  # num_neighbors=[5, 5],
                                  batch_size=512,
                                  shuffle=True,
                                  input_nodes=mask)

    model.eval()
    preds = []
    labels = []
    for batch in tqdm(test_loader):
        subgraphs = get_subgraph_snapshots(batch)
        subgraphs = [subgraph.to(device) for subgraph in subgraphs]
        batch = subgraphs[-1]
        batch_size = batch.batch_size
        out = model(subgraphs)[:batch_size]
        preds.append(out[:batch_size].argmax(1).detach())
        labels.append(batch.y[:batch_size].detach())
    preds = torch.cat(preds)
    labels = torch.cat(labels)
    score = []
    score += [metrics.f1_score(preds.cpu(), labels.cpu(), average='micro')]
    score += [metrics.f1_score(preds.cpu(), labels.cpu(), average='macro')]
    return score


best_val = -1e5
best_test = -1e5
best_metric_macros = None

start_time = time.time()
for epoch in tqdm(range(1, config['epochs']+1)):
    loss = train(snapshots)
    val_mif1, val_maf1 = test(snapshots, data.val_mask)
    test_mif1, test_maf1 = test(snapshots, data.test_mask)
    if best_val < val_mif1:
        best_val = val_mif1
        best_micro_f1 = test_mif1
        best_macro_f1 = test_maf1
        
    logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    logger.info(f'Micro-F1 - Val: {val_mif1:.2%}, Test: {test_mif1:.2%}, Best: {best_micro_f1:.2%}')
    logger.info(f'Macro-F1 - Val: {val_maf1:.2%}, Test: {test_maf1:.2%}, Best: {best_macro_f1:.2%}')
end_tim = time.time()

logger.warning(f'{args.dataset} - Best Micro-F1: {best_micro_f1:.2%} Best Macro-F1: {best_macro_f1:.2%} Time: {end_tim-start_time:.2f}s')

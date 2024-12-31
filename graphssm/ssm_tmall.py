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


from graphssm.transforms import RandomNodeSplit, ToTemporalUndirected, StratifyNodeSplit
from graphssm.datasets import DBLP, STARDataset, Tmall

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--dataset', type=str, default="Tmall")
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.01)
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
args.test_size = 1 - args.train_size
args.train_size = args.train_size - args.val_size

config = {
    "train_size": args.train_size,
    'hidden_channels': args.hidden_channels,
    'learning_rate': args.learning_rate,
    "weight_decay": args.weight_decay,
    "ssm_format": args.ssm_format,
    "token_mixer": args.token_mixer,
    'epochs': args.epochs,
    "seed": args.seed
}


root = './data'

transform = Compose(
    [ToTemporalUndirected(),
     StratifyNodeSplit(num_val=args.val_size, num_test=args.test_size, unknown=-1)])
data = Tmall(root=path, transform=transform, force_reload=False)[0]
device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

def to_data(temporal_data):
    x = temporal_data.x
    return Data(x=x,
                edge_index=temporal_data.edge_index,
                y=temporal_data.y)


bins = data.time_stamps
snapshots = [data.snapshot(end=i, last_node_attr=True)
             for i in range(data.num_snapshots)]
snapshots = [to_data(snapshot) for snapshot in snapshots]


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
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
train_loader = NeighborLoader(data.to_static(),
                              # num_neighbors=[-1, -1],
                              num_neighbors=[5, 5],
                              batch_size=2048,
                              shuffle=True,
                              input_nodes=data.train_mask)


num_nodes = data.x.size(0)
print(f'#Paras {sum(p.numel()/1e3 for p in model.parameters()):.3f}K')


def get_subgraph_snapshots(batch):
    edge_index = batch.edge_index
    edge_time = batch.edge_time
    subgraphs = []
    for i, t in enumerate(bins[:-1]):
        mask = edge_time <= t
        g = Data(x=batch.x[:, i, :], edge_index=edge_index[:, mask])
        subgraphs.append(g)
    batch.x = batch.x[:, -1, :]
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
def test(snapshots):
    model.eval()
    snapshots = [copy(snapshot).to(device) for snapshot in snapshots]
    pred = model(snapshots).argmax(dim=-1)
    metric_macros = []
    metric_micros = []
    for mask in [data.val_mask, data.test_mask]:
        if mask.sum() == 0:
            metric_macros.append(0)
            metric_micros.append(0)
        else:
            metric_macros.append(metrics.f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro'))
            metric_micros.append(metrics.f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='micro'))
    return metric_macros, metric_micros


best_val = -1e5
best_test = -1e5

start_time = time.time()
for epoch in tqdm(range(1, config['epochs']+1)):
    loss = train(snapshots)
    metric_macros, metric_micros = test(snapshots)
    val_mif1, test_mif1 = metric_micros
    val_maf1, test_maf1 = metric_macros
    if best_val < val_mif1:
        best_val = val_mif1
        best_micro_f1 = test_mif1
        best_macro_f1 = test_maf1
        
    logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    logger.info(f'Micro-F1 - Val: {val_mif1:.2%}, Test: {test_mif1:.2%}, Best: {best_micro_f1:.2%}')
    logger.info(f'Macro-F1 - Val: {val_maf1:.2%}, Test: {test_maf1:.2%}, Best: {best_macro_f1:.2%}')
end_tim = time.time()

logger.warning(f'{args.dataset} - Best Micro-F1: {best_micro_f1:.2%} Best Macro-F1: {best_macro_f1:.2%} Time: {end_tim-start_time:.2f}s')

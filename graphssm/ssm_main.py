import time
import argparse
import torch
import os.path as osp
import numpy as np
from tqdm import tqdm
from sklearn import metrics
from torch_geometric import seed_everything
from torch_geometric.transforms import Compose, NormalizeFeatures
import torch.nn.functional as F
from loguru import logger

from graphssm.transforms import RandomNodeSplit, ToTemporalUndirected, StratifyNodeSplit
from graphssm.datasets import DBLP, Tmall, STARDataset

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")

parser = argparse.ArgumentParser()
# ['DBLP', 'dblp3', 'dblp5', 'reddit', 'brain']:
parser.add_argument('--seed', type=int, default=2024)
parser.add_argument('--dataset', type=str, default="dblp3")
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--d_state', type=int, default=16)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--learning_rate', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--ssm_format', type=str, default='siso')
parser.add_argument('--token_mixer', type=str, default='interp')
parser.add_argument('--train_size', type=float, default=0.8)
parser.add_argument('--val_size', type=float, default=0.05)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--model_name', type=str, default="ssm")
args = parser.parse_args()
logger.warning(args)

seed_everything(args.seed)

if args.dataset in ['dblp3', 'dblp5', 'reddit', 'brain']:
    # according to STAR: Spatio-Temporal Attentive RNN for Node Classification in Temporal Attributed Graphs
    args.test_size = 0.1
    args.train_size = 0.81
    args.val_size = 0.09
else:
    args.test_size = 1 - args.train_size
    args.train_size = args.train_size - args.val_size

if args.model_name == "ssm":
    from models.ssm import DiagonalSSM
elif args.model_name == "s6":
    from models.s6 import DiagonalS6SSM as DiagonalSSM

config = {
    "dataset": args.dataset,
    "train_size": args.train_size,
    'hidden_channels': args.hidden_channels,
    'learning_rate': args.learning_rate,
    "weight_delay": args.weight_decay,
    "ssm_format": args.ssm_format,
    "token_mixer": args.token_mixer,
    'epochs': args.epochs,
    "d_state": args.d_state
}
transform = Compose(
    [ToTemporalUndirected(),
     StratifyNodeSplit(num_val=args.val_size, num_test=args.test_size, unknown=-1)])   
dataset = args.dataset

logger.warning('Loading dataset...')
if dataset == 'dblp10':
    path = './data/'
    data = DBLP(root=path, transform=transform, force_reload=False)[0]
elif dataset in ['dblp3', 'dblp5', 'reddit', 'brain']:
    path = './data/'
    data = STARDataset(root=path, name=dataset,
                       transform=transform, force_reload=False)[0]

device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

logger.warning('To snapshots...')
snapshots = [data.snapshot(start=i) for i in range(data.num_snapshots)]
snapshots = [snapshot.to(device) for snapshot in snapshots]

model = DiagonalSSM(
    data.x.size(-1),
    data.y.max().item()+1,
    hidden_channels=config["hidden_channels"],
    ssm_format=config["ssm_format"],
    token_mixer=config["token_mixer"],
    d_state=config['d_state'],
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=args.weight_decay)
logger.warning(f'#Paras {sum(p.numel()/1e3 for p in model.parameters()):.3f}K')

# data.train_mask = data.train_mask | data.val_mask
# print(data.train_mask.float().mean())
# print(data.val_mask.float().mean())
# print(data.test_mask.float().mean())

def train():
    model.train()
    optimizer.zero_grad()
    out = model(snapshots)
    data = snapshots[-1]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def test():
    model.eval()
    pred = model(snapshots).argmax(dim=-1)
    data = snapshots[-1]
    metric_macros = []
    metric_micros = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        if mask.sum() == 0:
            metric_macros.append(0)
            metric_micros.append(0)
        else:
            metric_macros.append(metrics.f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='macro'))
            metric_micros.append(metrics.f1_score(data.y[mask].cpu(), pred[mask].cpu(), average='micro'))
    return metric_macros, metric_micros


best_val = -1e5
best_test = -1e5
logger.info(config)

start_time = time.time()
for epoch in tqdm(range(1, config['epochs']+1)):
    loss = train()
    metric_macros, metric_micros = test()
    train_mif1, val_mif1, test_mif1 = metric_micros
    train_maf1, val_maf1, test_maf1 = metric_macros
    if best_val < val_mif1:
        best_val = val_mif1
        best_micro_f1 = test_mif1
        best_macro_f1 = test_maf1
        
    if epoch % 20 == 0:
        logger.info(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        logger.info(f'Micro-F1: Train: {train_mif1:.2%}, Val: {val_mif1:.2%}, Test: {test_mif1:.2%}, Best: {best_micro_f1:.2%}')
        logger.info(f'Macro-F1: Train: {train_maf1:.2%}, Val: {val_maf1:.2%}, Test: {test_maf1:.2%}, Best: {best_macro_f1:.2%}')
end_tim = time.time()

logger.warning(f'{dataset} - Best Micro-F1: {best_micro_f1:.2%} Best Macro-F1: {best_macro_f1:.2%} Time: {end_tim-start_time:.2f}s')

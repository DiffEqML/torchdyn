import os.path as osp

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SplineConv
from torchdyn.models import NeuralDE

dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'datasets', dataset)
dataset = Planetoid(path, dataset, transform=T.TargetIndegree())
data = dataset[0]

data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[:data.num_nodes - 1000] = 1
data.val_mask = None
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask[data.num_nodes - 500:] = 1


class GCNLayer(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNLayer, self).__init__()

        if input_size != output_size:
            raise AttributeError('input size must equal output size')

        self.conv1 = SplineConv(input_size, output_size, dim=1, kernel_size=2).to(device)
        self.conv2 = SplineConv(input_size, output_size, dim=1, kernel_size=2).to(device)

    def forward(self, x):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.func = GCNLayer(input_size=64, output_size=64)

        self.conv1 = SplineConv(dataset.num_features, 64, dim=1, kernel_size=2).to(device)
        self.neuralDE = NeuralDE(self.func, solver='rk4', s_span=torch.linspace(0, 1, 3)).to(device)
        self.conv2 = SplineConv(64, dataset.num_classes, dim=1, kernel_size=2).to(device)

    def forward(self, x):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        x = F.tanh(self.conv1(x, edge_index, edge_attr))
        x = F.dropout(x, training=self.training)
        x = self.neuralDE(x)
        x = F.tanh(self.conv2(x, edge_index, edge_attr))

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-3)


def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model(data.x)[data.train_mask], data.y[data.train_mask]).backward()
    optimizer.step()


def test():
    model.eval()
    logits, accs = model(data.x), []
    for _, mask in data('train_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs


for epoch in range(1, 201):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    print(log.format(epoch, *test()))

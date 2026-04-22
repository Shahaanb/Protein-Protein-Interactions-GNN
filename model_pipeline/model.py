import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class ProteomeXGAT(torch.nn.Module):
    def __init__(self, in_channels=64, hidden_channels=32, out_channels=1, heads=4):
        super(ProteomeXGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, concat=False)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = F.elu(self.conv1(x, edge_index))
        x, (attention_edge_index, attention_weights) = self.conv2(x, edge_index, return_attention_weights=True)
        x = F.elu(x)
        
        # Aggregate attention to find hotspots (nodes that receive high attention)
        node_attention_scores = torch.zeros(x.size(0), device=x.device)
        target_nodes = attention_edge_index[1]
        weights_sum = attention_weights.sum(dim=-1).squeeze()
        
        # Scatter add to compute total attention received by each node
        node_attention_scores.scatter_add_(0, target_nodes, weights_sum)
        
        # Pool graph
        x_pooled = global_mean_pool(x, batch)
        out = self.lin(x_pooled)
        prob = torch.sigmoid(out)
        
        return prob, node_attention_scores

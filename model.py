import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class ResidualGCNLayer(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(ResidualGCNLayer, self).__init__()
        self.conv = pyg_nn.GCNConv(input_dim, output_dim)
        if input_dim != output_dim:
            self.residual = nn.Linear(input_dim, output_dim)
        else:
            self.residual = None

    def forward(self, x, edge_index):
        out = self.conv(x, edge_index)
        if self.residual is not None:
            x = self.residual(x)
        return F.relu(out + x)

class PredictMiddleViewResidualGCN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3):
        super(PredictMiddleViewResidualGCN, self).__init__()
        self.gcn_layers = nn.ModuleList([ResidualGCNLayer(input_dim if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)])
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        mask = data.hidden_mask
        
        x = torch.where(mask.unsqueeze(1).repeat(1, 4), x, torch.zeros_like(x))
        
        for gcn in self.gcn_layers:
            x = gcn(x, edge_index)
        
        x_out = self.fc(x)

        return x_out, mask

import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    """ 简单的图卷积层 """ #
    def __init__(self, input_features, output_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.randn(input_features, output_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.randn(output_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_normal_(self.weight.data)
        nn.init.zeros_(self.bias.data)

    def forward(self, text, adjective):
        hidden = torch.matmul(text, self.weight)
        denominator = torch.sum(adjective, dim=-1, keepdim=True) + 1
        output = torch.matmul(adjective, hidden) / denominator
        if self.bias is not None:
            return output + self.bias
        else:
            return output



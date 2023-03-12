import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# code taken from https://github.com/ykasten/layered-neural-atlases


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def positionalEncoding_vec(in_tensor, b):
    proj = torch.einsum("ij, k -> ijk", in_tensor, b)  # shape (batch, in_tensor.size(1), freqNum)
    mapped_coords = torch.cat((torch.sin(proj), torch.cos(proj)), dim=1)  # shape (batch, 2*in_tensor.size(1), freqNum)
    output = mapped_coords.transpose(2, 1).contiguous().view(mapped_coords.size(0), -1)
    return output


class IMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim=256,
        use_positional=True,
        positional_dim=10,
        skip_layers=[4, 6],
        num_layers=8,  # includes the output layer
        verbose=True,
        use_tanh=True,
        apply_softmax=False,
    ):
        super(IMLP, self).__init__()
        self.verbose = verbose
        self.use_tanh = use_tanh
        self.apply_softmax = apply_softmax
        if apply_softmax:
            self.softmax = nn.Softmax()
        if use_positional:
            encoding_dimensions = 2 * input_dim * positional_dim
            self.b = torch.tensor([(2 ** j) * np.pi for j in range(positional_dim)], requires_grad=False)
        else:
            encoding_dimensions = input_dim

        self.hidden = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                input_dims = encoding_dimensions
            elif i in skip_layers:
                input_dims = hidden_dim + encoding_dimensions
            else:
                input_dims = hidden_dim

            if i == num_layers - 1:
                # last layer
                self.hidden.append(nn.Linear(input_dims, output_dim, bias=True))
            else:
                self.hidden.append(nn.Linear(input_dims, hidden_dim, bias=True))

        self.skip_layers = skip_layers
        self.num_layers = num_layers

        self.positional_dim = positional_dim
        self.use_positional = use_positional

        if self.verbose:
            print(f"Model has {count_parameters(self)} params")

    def forward(self, x):
        if self.use_positional:
            if self.b.device != x.device:
                self.b = self.b.to(x.device)
            pos = positionalEncoding_vec(x, self.b)
            x = pos

        input = x.detach().clone()
        for i, layer in enumerate(self.hidden):
            if i > 0:
                x = F.relu(x)
            if i in self.skip_layers:
                x = torch.cat((x, input), 1)
            x = layer(x)
        if self.use_tanh:
            x = torch.tanh(x)

        if self.apply_softmax:
            x = self.softmax(x)
        return x

import torch
import torch.nn as nn


class RNNCell(nn.Module):
    """
    A single RNN cell that processes one timestep of input.
    Implements: h_t = tanh(W_hh * h_{t-1} + W_xh * x_t)
                y_t = W_hy * h_t
    """
    def __init__(self, inp_size, hidden_size, nonlinear=nn.Tanh(), bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.L_xh = nn.Linear(inp_size, hidden_size, bias=bias)
        self.L_hh = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.L_hy = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.nonlinear = nonlinear

    def forward(self, x, hidden=None):
        """
        Forward pass through RNN cell.

        Args:
            x: [batch, input_size] - Input at current timestep
            hidden: [batch, hidden_size] - Hidden state from previous timestep

        Returns:
            out: [batch, hidden_size] - Output at current timestep
            hidden: [batch, hidden_size] - Updated hidden state
        """
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        hidden = self.nonlinear(self.L_hh(hidden) + self.L_xh(x))
        out = self.L_hy(hidden)
        return out, hidden


class RNN(nn.Module):
    """
    Multi-layer RNN that stacks multiple RNN cells.
    Processes sequences of inputs through multiple layers with optional dropout.
    """

    def __init__(self, inp_size, hidden_size, no_of_layers, dropout=0.5):
        super().__init__()
        self.no_of_layers = no_of_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        for l in range(self.no_of_layers):
            if l == 0:
                self.layers.append(RNNCell(inp_size, hidden_size))
            else:
                self.layers.append(RNNCell(hidden_size, hidden_size))

    def forward(self, x, hidden=None):
        """
        Forward pass through all RNN layers for entire sequence.

        Args:
            x: [batch, timesteps, input_size] - Input sequence
            hidden: list of [batch, hidden_size] tensors for each layer or None

        Returns:
            outputs: [batch, timesteps, hidden_size] - Output sequence
            hidden: list of final hidden states for each layer
        """
        batch, timesteps, _ = x.shape

        if hidden is None:
            hidden = [None] * self.no_of_layers

        outputs = []
        for t in range(timesteps):
            x_t = x[:, t, :]
            for l, rnn in enumerate(self.layers):
                x_t, hidden[l] = rnn(x_t, hidden[l])
                if l < self.no_of_layers - 1:
                    x_t = self.dropout(x_t)  # better ignore dropout on last layer
            outputs.append(x_t)

        outputs = torch.stack(outputs, dim=1)
        return outputs, hidden


class CharRNN(nn.Module):
    """
    Character-level RNN for sequence modeling.
    Takes character embeddings and predicts next character probabilities.
    """

    def __init__(self, inp_size, hidden_size, no_of_layers, dropout=0.1):
        super().__init__()
        self.rnn = RNN(inp_size, hidden_size, no_of_layers, dropout)
        self.fc = nn.Linear(hidden_size, inp_size)

    def forward(self, x, hidden=None):
        """
        Forward pass through character RNN.

        Args:
            x: [batch, timesteps, input_size] - One-hot encoded character sequence
            hidden: list of [batch, hidden_size] tensors for each layer or None

        Returns:
            out: [batch, timesteps, input_size] - Logits for next character prediction
            hidden: list of final hidden states for each layer
        """
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)
        return out, hidden


if __name__ == "__main__":
    batch = 1
    timesteps = 10
    vocab_size = 26
    hidden_size = 64
    no_of_layers = 3
    dropout = 0.1
    model = CharRNN(vocab_size, hidden_size, no_of_layers, dropout)
    x = torch.randn(batch, timesteps, vocab_size)
    out, hidden = model(x)
    print(out.shape, len(hidden), hidden[0].shape)

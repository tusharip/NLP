import torch
import torch.nn as nn


class ForgetGate(nn.Module):
    def __init__(self, inp_size, hidden_size):
        super().__init__()
        self.L_xh = nn.Linear(inp_size, hidden_size, bias=True)
        self.L_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        """Forward pass through forget gate.
        f_t = sigmoid(W_xh * x_t + W_hh * h_{t-1} + b_f)

        Args:
            x : [batch, input_size] - Input at current timestep
            hidden: [batch, hidden_size] - Hidden state from previous timestep
        Returns:
        f_t: [batch, hidden_size]
        """
        f_t = self.sigmoid(self.L_xh(x) + self.L_hh(hidden))
        return f_t


class InputGate(nn.Module):
    """
    Input gate for LSTM cell.
    part1 input : i_t = sigmoid(W_xhi * x_t + W_hhi * h_{t-1} + b_i)
    part2 candidate : c_t = tanh(W_xhc * x_t + W_hhc * h_{t-1} + b_c)
    """

    def __init__(self, inp_size, hidden_size):
        super().__init__()
        self.L_xhi = nn.Linear(inp_size, hidden_size, bias=True)
        self.L_hhi = nn.Linear(hidden_size, hidden_size, bias=False)
        self.L_xhc = nn.Linear(inp_size, hidden_size, bias=True)
        self.L_hhc = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden):
        """Forward pass through input gate.
        Args:
            x : [batch, input_size] - Input at current timestep
            hidden: [batch, hidden_size] - Hidden state from previous timestep
        Returns:
        i_t: [batch, hidden_size]
        c_t: [batch, hidden_size]
        """
        i_t = self.sigmoid(self.L_xhi(x) + self.L_hhi(hidden))
        c_t = self.tanh(self.L_xhc(x) + self.L_hhc(hidden))
        return i_t, c_t


class OutputGate(nn.Module):
    """Output gate for LSTM cell.
    o_t = sigmoid(W_xho * x_t + W_hho * h_{t-1} + b_o)
    h_t = o_t * tanh(c_t)
    """

    def __init__(self, inp_size, hidden_size):
        super().__init__()
        self.L_xho = nn.Linear(inp_size, hidden_size, bias=True)
        self.L_hho = nn.Linear(hidden_size, hidden_size, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, hidden, cell):
        """Forward pass through output gate.
        Args:
            x : [batch, input_size] - Input at current timestep
            hidden: [batch, hidden_size] - Hidden state from previous timestep
            cell: [batch, hidden_size] - Cell state from previous timestep
        Returns:
        o_t: [batch, hidden_size]
        h_t: [batch, hidden_size]
        """
        o_t = self.sigmoid(self.L_xho(x) + self.L_hho(hidden))
        h_t = o_t * self.tanh(cell)
        return h_t


class LSTMCell(nn.Module):
    """
    LSTM cell that processes one timestep of input.
    Implements: forget Gate, Input Gate, Cell update, output Gate
    """

    def __init__(self, inp_size, hidden_size):
        super().__init__()
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.forget_gate = ForgetGate(inp_size, hidden_size)
        self.input_gate = InputGate(inp_size, hidden_size)
        self.output_gate = OutputGate(inp_size, hidden_size)

    def forward(self, x, state=[None, None]):
        """Forward pass through LSTM cell.
        Args:
            x : [batch, input_size] - Input at current timestep
            state: tuple of (hidden, cell) - Hidden and cell states from previous timestep
        Returns:
        h_t: [batch, hidden_size] - Hidden state at current timestep
        cell: [batch, hidden_size] - Cell state at current timestep
        """
        hidden, cell = state
        if hidden is None:
            hidden = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        if cell is None:
            cell = torch.zeros(x.shape[0], self.hidden_size, device=x.device)
        f_t = self.forget_gate(x, hidden)
        i_t, c_t = self.input_gate(x, hidden)

        cell = f_t * cell + i_t * c_t  # update cell state from input and forget gate
        h_t = self.output_gate(x, hidden, cell)  # update hidden state from output gate
        return h_t, cell


class LSTM(nn.Module):
    """
    LSTM that processes a sequence of inputs.
    """

    def __init__(self, inp_size, hidden_size, no_of_layers, dropout=0.1):
        super().__init__()
        self.no_of_layers = no_of_layers
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        for l in range(self.no_of_layers):
            if l == 0:
                self.layers.append(LSTMCell(inp_size, hidden_size))
            else:
                self.layers.append(LSTMCell(hidden_size, hidden_size))

    def forward(self, x, state=[None, None]):
        """Forward pass through LSTM.
        Args:
            x : [batch, timesteps, input_size] - Input sequence
            state: tuple of (hidden, cell) - Hidden and cell states from previous timestep
        Returns:
        outputs: [batch, timesteps, hidden_size] - Output sequence
        hidden: [batch, hidden_size] * no_of_layers - Final hidden state
        cell: [batch, hidden_size] * no_of_layers - Final cell state
        """
        batch, timesteps, _ = x.shape
        hidden, cell = state
        if hidden is None:
            hidden = [None] * self.no_of_layers
        if cell is None:
            cell = [None] * self.no_of_layers
        outputs = []
        for t in range(timesteps):
            x_t = x[:, t, :]
            for l, lstm in enumerate(self.layers):
                state = [hidden[l], cell[l]]
                h_t,  cell[l] = lstm(x_t, state)
                hidden[l] = h_t
                x_t = h_t #pass to next layer
                if l < self.no_of_layers - 1:
                    x_t = self.dropout(x_t)  # better ignore dropout on last layer
            outputs.append(x_t)
        outputs = torch.stack(outputs, dim=1)
        return outputs, [hidden, cell]


class CharLSTM(nn.Module):
    """
    Character-level LSTM for sequence modeling.
    Takes character embeddings and predicts next character probabilities.
    """

    def __init__(self, inp_size, hidden_size, no_of_layers, dropout=0.1):
        super().__init__()
        self.lstm = LSTM(inp_size, hidden_size, no_of_layers, dropout)
        self.fc = nn.Linear(hidden_size, inp_size)

    def forward(self, x, hidden=[None, None]):
        """
        Forward pass through character LSTM.
        Args:
            x: [batch, timesteps, input_size] - One-hot encoded character sequence
            hidden: tuple of (hidden, cell) - Hidden and cell states from previous timestep
        Returns:
        outputs: [batch, timesteps, input_size] - Logits for next character prediction
        hidden: tuple of (hidden, cell) - Final hidden and cell states
        """
        out, state = self.lstm(x, hidden)
        out = self.fc(out)
        return out, state


if __name__ == "__main__":
    batch = 1
    timesteps = 10
    vocab_size = 26
    hidden_size = 64
    no_of_layers = 3
    dropout = 0.1
    model = CharLSTM(vocab_size, hidden_size, no_of_layers, dropout)
    x = torch.randn(batch, timesteps, vocab_size)
    out, (h, c) = model(x)
    print(out.shape, h[0].shape, c[0].shape, len(h), len(c))

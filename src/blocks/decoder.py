import torch
import torch.nn as nn

from src.blocks.common import StackedLSTM


class RnnDecoder(nn.Module):
    def __init__(self, tgt_vocab_size, hidden_size):
        super(RnnDecoder, self).__init__()

        self.hidden_size = hidden_size
        dropout_prob = 0.2
        num_layers = 1

        self.rnn = StackedLSTM(input_size=2 * self.hidden_size,
                               hidden_size=self.hidden_size,
                               num_layers=num_layers,
                               dropout=dropout_prob)

        self.inner_hidden_size = 768
        self.W_d = nn.Linear(self.hidden_size, self.inner_hidden_size)
        self.V_d = nn.Linear(self.hidden_size, self.inner_hidden_size)
        self.W_o = nn.Linear(self.inner_hidden_size, tgt_vocab_size)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input, state, c_t, prev_predicted_labels=None):
        output, state = self.rnn(input, state)
        return output, state

    def compute_score(self, hiddens, c_t, prev_predicted_labels=None, use_softmax=False):
        scores = self.W_o(self.activation(self.W_d(hiddens) + self.V_d(c_t)))
        I = torch.zeros_like(scores)
        if prev_predicted_labels:
            for predicted_labels in prev_predicted_labels:
                I[(list(range(I.size(0))), predicted_labels)] = -1 * float('inf')
        scores = scores + I
        if use_softmax:
            scores = self.softmax(scores)
        return scores


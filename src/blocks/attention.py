import torch
import torch.nn as nn


class SgmAttention(nn.Module):
    def __init__(self, encoder_hidden_size, decoder_hidden_size, att_hidden_size):
        super(SgmAttention, self).__init__()
        self.U = nn.Linear(encoder_hidden_size, att_hidden_size)
        self.W = nn.Linear(decoder_hidden_size, att_hidden_size)
        self.tanh = nn.Tanh()
        self.V = nn.Linear(att_hidden_size, 1)
        self.softmax = nn.Softmax(dim=-1)

    def init_context(self, context):
        self.context = context

    def forward(self, s):
        state_term = self.W(s).unsqueeze(1)
        context_term = self.U(self.context)
        sum_activation = self.tanh(context_term + state_term.expand_as(context_term))
        weights = self.V(sum_activation).squeeze(-1)
        softmax_weights = self.softmax(weights)
        c_t = torch.bmm(softmax_weights.unsqueeze(1), self.context).squeeze(1)
        return c_t

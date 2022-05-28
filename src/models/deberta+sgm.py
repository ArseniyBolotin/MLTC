import torch
import torch.nn as nn

from src.blocks.encoder import DebertaEncdoer
from src.blocks.attention import SgmAttention
from src.blocks.decoder import RnnDecoder

from src.utils.constants import BOS, PAD, EOS


class BertSGM(nn.Module):
    def __init__(self, device):
        super(BertSGM, self).__init__()
        self.device = device
        tgt_vocab_size = 58
        tgt_embedding_size = 768
        self.decoder_hidden_size = 768
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, tgt_embedding_size)
        self.encoder = DebertaEncdoer()
        encoder_hidden_size = self.encoder.hidden_size
        self.mapping = nn.Linear(encoder_hidden_size, self.decoder_hidden_size)
        self.attention = SgmAttention(encoder_hidden_size=encoder_hidden_size,
                                      decoder_hidden_size=self.decoder_hidden_size, att_hidden_size=768)
        self.decoder = RnnDecoder(tgt_vocab_size=tgt_vocab_size, hidden_size=self.decoder_hidden_size,
                                  input_size=encoder_hidden_size + tgt_embedding_size, encoder_hidden_size=encoder_hidden_size)
        self.criterion = self.create_criterion(tgt_vocab_size)

    def forward(self, src, tgt):
        context, encoder_state = self.encoder(src)

        self.attention.init_context(context)
        y_hats = self.tgt_embedding(tgt[:, :-1])

        batch_size = y_hats.size(0)
        prev_predicted_labels = []
        saved_scores = []
        decoder_state = (self.mapping(
            encoder_state[0]), self.mapping(encoder_state[1]))

        for y_hat, t in zip(y_hats.split(1, dim=1), tgt[:, 1:].transpose(0, 1)):
            c_t = self.attention(decoder_state[0].squeeze(0))
            input = torch.cat([y_hat.squeeze(1), c_t], dim=-1)
            output, decoder_state = self.decoder(input, decoder_state, c_t)
            scores = self.decoder.compute_score(
                output, c_t, prev_predicted_labels)
            saved_scores.append(scores)
            prev_predicted_labels.append(t)

        scores = torch.stack(saved_scores).transpose(0, 1)
        return self.compute_loss(scores, tgt)

    def compute_loss(self, scores, tgt):
        loss = 0.
        for score, t in zip(scores, tgt[:, 1:]):
            loss += self.criterion(score, t)
        return loss / tgt.size(0)

    def create_criterion(self, tgt_vocab_size):
        weight = torch.ones(tgt_vocab_size)
        weight[PAD] = 0
        crit = nn.CrossEntropyLoss(weight, ignore_index=PAD)
        return crit

    def predict(self, src, max_steps=10):
        context, encoder_state = self.encoder(src)
        self.attention.init_context(context)
        batch_size = src['input_ids'].size(0)
        y_hat = self.tgt_embedding(torch.tensor(
            [BOS for _ in range(batch_size)]).to(self.device))
        decoder_state = (self.mapping(
            encoder_state[0]), self.mapping(encoder_state[1]))

        predicted_labels = []
        eos_predicted = torch.tensor(
            [False for _ in range(batch_size)]).to(self.device)

        for _ in range(max_steps):
            c_t = self.attention(decoder_state[0].squeeze(0))
            input = torch.cat([y_hat.squeeze(1), c_t], dim=-1)
            output, decoder_state = self.decoder(input, decoder_state, c_t)
            scores = self.decoder.compute_score(output, c_t, predicted_labels)
            prediction = torch.argmax(scores, dim=-1)
            y_hat = self.tgt_embedding(prediction.to(self.device))
            predicted_labels.append(prediction.tolist())
            eos_predicted = eos_predicted | (prediction == EOS)
            if torch.all(eos_predicted):
                break

        return torch.tensor(predicted_labels)

from abc import ABCMeta, abstractmethod
import torch
from transformers import AutoModel, logging
import torch
import torch.nn as nn

logging.set_verbosity_error()


class BaseEncoder(nn.Module, metaclass=ABCMeta):
    @property
    @abstractmethod
    def hidden_size(self):
        pass

    @abstractmethod
    def forward(self, inputs):
        pass


class TransformerEncoder(BaseEncoder):
    def __init__(self, model_type):
        super(TransformerEncoder, self).__init__()
        self.model = AutoModel.from_pretrained(model_type)


class BertEncdoer(TransformerEncoder):
    def __init__(self, model_type="bert-base-uncased"):
        super(BertEncdoer, self).__init__(model_type)
        self.dropout_prob = 0.2
        self.dropout = nn.Dropout(self.dropout_prob)

    hidden_size = 768

    def forward(self, inputs):
        bert_output = self.bert_model(**inputs)
        pooler_output = self.dropout(bert_output.pooler_output)
        return bert_output.last_hidden_state, (torch.unsqueeze(pooler_output, 0), torch.unsqueeze(pooler_output, 0))


class DebertaEncdoer(TransformerEncoder):
    def __init__(self, model_type="microsoft/deberta-v3-large"):
        super(DebertaEncdoer, self).__init__(model_type)

    hidden_size = 1024

    def forward(self, inputs):
        output = self.model(**inputs)
        return output.last_hidden_state, (output.last_hidden_state.transpose(0, 1)[:1], output.last_hidden_state.transpose(0, 1)[:1])

import torch
import torch.nn as nn

'''
Implements a simple LSTM Tagging model.
'''

class LSTMTagger(nn.Module):
    def __init__(self, word_vocab_sz, tag_vocab_sz, embedding_dim, hidden_dim, dropout=0.5, bidirectional=True):
        super(LSTMTagger, self).__init__()
        self.embedding = nn.Embedding(word_vocab_sz, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional)
        self.fc1 = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, tag_vocab_sz)
        self.dropout = nn.Dropout(dropout)

    def init_hidden(self, bs):
        params = next(self.parameters())
        hidden_dim = self.rnn.hidden_size
        directions = 2 if self.rnn.bidirectional else 1

        hidden = params.new_zeros(directions, bs, hidden_dim)
        cell = params.new_zeros(directions, bs, hidden_dim)
        return hidden, cell

    def forward(self, x):
        msl, bs = x.shape
        out = self.embedding(x)
        hidden, cell = self.init_hidden(bs)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))

        out = self.dropout(out)
        out = self.fc1(out)
        return out
